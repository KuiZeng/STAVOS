import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


def aggregate_objects(pred_seg, object_ids):  # pred_seg 为预测的分割结果，object_ids 为物体的 ID
    bg_seg, _ = torch.stack([seg[:, 0, :, :] for seg in pred_seg.values()], dim=1).min(dim=1)  # 将背景的分割结果取最小值，得到背景分割结果
    bg_seg = torch.stack([1 - bg_seg, bg_seg], dim=1)  # 构造一个二分类的背景分割结果
    logit = {n: seg[:, 1:, :, :].clamp(1e-7, 1 - 1e-7) / seg[:, 0, :, :].clamp(1e-7, 1 - 1e-7) for n, seg in [(-1, bg_seg)] + list(pred_seg.items())}  # 计算每个物体和背景的对数几率
    logit_sum = torch.cat(list(logit.values()), dim=1).sum(dim=1, keepdim=True)  # 计算所有物体和背景的对数几率之和
    aggregated_lst = [logit[n] / logit_sum for n in [-1] + object_ids]  # 对每个物体和背景的对数几率进行归一化

    # 将每个物体和背景的归一化后的对数几率串联起来，得到最终的聚合结果
    aggregated_inv_lst = [1 - elem for elem in aggregated_lst]
    aggregated = torch.cat([elem for lst in zip(aggregated_inv_lst, aggregated_lst) for elem in lst], dim=-3)

    # 根据每个像素的最大值，将其分配给相应的物体或背景
    mask_tmp = aggregated[:, 1::2, :, :].argmax(dim=-3, keepdim=True)
    pred_mask = torch.zeros_like(mask_tmp)
    for idx, obj_idx in enumerate(object_ids):
        pred_mask[mask_tmp == (idx + 1)] = obj_idx

    return pred_mask, {obj_idx: aggregated[:, 2 * (idx + 1):2 * (idx + 2), :, :] for idx, obj_idx in enumerate(object_ids)}  # 返回聚合后的分割结果和每个物体的对数几率


# 定义一个函数，计算输入图像的填充量，使得图像的高度和宽度能够被div整除
def get_padding(h, w, div):
    h_pad = (div - h % div) % div  # 计算高度上的填充量
    w_pad = (div - w % div) % div  # 计算宽度上的填充量
    padding = [(w_pad + 1) // 2, w_pad // 2, (h_pad + 1) // 2, h_pad // 2]  # 将宽度和高度的填充量划分到四个元素中
    return padding


# 定义一个函数，将给定的图像进行填充，并返回填充后的图像和对应的mask
def attach_padding(imgs, given_masks, padding):
    B, L, C, H, W = imgs.size()  # 获取输入图像的基本信息
    imgs = imgs.view(B * L, C, H, W)  # 对输入的图像进行展开，方便进行填充
    imgs = F.pad(imgs, padding, mode='reflect')  # 进行填充操作，采用反射模式填充
    _, _, height, width = imgs.size()  # 更新填充后的高度和宽度
    imgs = imgs.view(B, L, C, height, width)  # 将填充后的图像进行reshape为原来的维度
    given_masks = [F.pad(label.float(), padding, mode='reflect').long() if label is not None else None for label in given_masks]  # 对给定的mask进行填充
    return imgs, given_masks


# 定义一个函数，将输出图像去掉填充部分，并返回去掉填充后的图像
def detach_padding(output, padding):
    if isinstance(output, list):  # 如果输出是个list，则对其内部每个元素都进行去填充的操作
        return [detach_padding(x, padding) for x in output]
    else:
        _, _, _, height, width = output.size()  # 获取输出图像的基本信息
        return output[:, :, :, padding[2]:height - padding[3], padding[0]:width - padding[1]]  # 去除填充部分


# 基础模块
class Conv(nn.Sequential):  # 定义一个卷积层的类，继承自nn.Sequential
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))  # 加入一个卷积层
        for m in self.children():  # 对该层的每个子模块进行操作
            if isinstance(m, nn.Conv2d):  # 如果是卷积层，则对其权重进行初始化
                nn.init.kaiming_uniform_(m.weight)  # 使用kaiming_uniform_初始化该层的权重
                if m.bias is not None:  # 如果存在偏置项，则将其设为0
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):  # 定义一个卷积层+ReLU激活函数的类，继承自nn.Sequential
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))  # 加入一个卷积层
        self.add_module('relu', nn.ReLU())  # 加入一个ReLU激活函数层
        for m in self.children():  # 对该层的每个子模块进行操作
            if isinstance(m, nn.Conv2d):  # 如果是卷积层，则对其权重进行初始化
                nn.init.kaiming_uniform_(m.weight)  # 使用kaiming_uniform_初始化该层的权重
                if m.bias is not None:  # 如果存在偏置项，则将其设为0
                    nn.init.constant_(m.bias, 0)


class DeConv(nn.Sequential):  # 定义一个反卷积层的类，继承自nn.Sequential
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('deconv', nn.ConvTranspose2d(*conv_args))  # 加入一个反卷积层
        for m in self.children():  # 对该层的每个子模块进行操作
            if isinstance(m, nn.ConvTranspose2d):  # 如果是反卷积层，则对其权重进行初始化
                nn.init.kaiming_uniform_(m.weight)  # 使用kaiming_uniform_初始化该层的权重
                if m.bias is not None:  # 如果存在偏置项，则将其设为0
                    nn.init.constant_(m.bias, 0)


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = Conv(c, c, 3, 1, 1)  # 建立一个3x3的卷积层
        self.conv2 = nn.Sequential(ConvRelu(c, c, 1, 1, 0), Conv(c, c, 1, 1, 0))  # 连接两个卷积层，第一个是带有ReLU激活函数的1x1卷积层，第二个是1x1的卷积层
        self.conv3 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))  # 连接两个卷积层，第一个是带有ReLU激活函数的3x3卷积层，第二个是3x3的卷积层

    def forward(self, x):
        x = self.conv1(x)  # 将输入通过第一个卷积层进行卷积处理
        c = torch.sigmoid(self.conv2(F.adaptive_avg_pool2d(x, output_size=(1, 1))) + self.conv2(
            F.adaptive_max_pool2d(x, output_size=(1, 1))))  # 对通过自适应平均池化和自适应最大池化后的x经过两个1x1卷积层得到的向量求sigmoid函数，得到权重c
        x = x * c  # 将x乘上权重c
        s = torch.sigmoid(self.conv3(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]],
                                               dim=1)))  # 将x沿着通道维度求均值和最大值，得到两个信息，通过拼接后，通过带有ReLU激活函数的3x3卷积层，然后再通过一个不带ReLU激活函数的3x3卷积层得到权重s
        x = x * s  # 将x乘上权重s
        return x  # 返回处理后的结果


# 编码模块
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # backbone = tv.models.densenet121(pretrained=True).features  # 建立预训练的densenet121网络，用于作为Encoder的主干网络
        backbone = tv.models.densenet201(pretrained=True).features  # 建立预训练的densenet201网络，用于作为Encoder的主干网络

        self.conv0 = backbone.conv0  # 获取第一个卷积层
        self.norm0 = backbone.norm0  # 获取第一个标准化层
        self.relu0 = backbone.relu0  # 获取第一个ReLU激活函数层
        self.pool0 = backbone.pool0  # 获取第一个池化层
        self.denseblock1 = backbone.denseblock1  # 获取第一个dense block
        self.transition1 = backbone.transition1  # 获取第一个过渡层
        self.denseblock2 = backbone.denseblock2  # 获取第二个dense block
        self.transition2 = backbone.transition2  # 获取第二个过渡层
        self.denseblock3 = backbone.denseblock3  # 获取第三个dense block
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # 将均值[0.485, 0.456, 0.406]转换成张量，并注册为模型的buffer
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))  # 将标准差[0.229, 0.224, 0.225]转换成张量，并注册为模型的buffer

    def forward(self, img):
        x = (img - self.mean) / self.std  # 根据均值和标准差对输入进行标准化
        # print(x.size())     # torch.Size([2, 3, 384, 384])
        x = self.conv0(x)  # 对输入进行卷积操作
        # print(x.size())     # torch.Size([2, 64, 192, 192])
        x = self.norm0(x)  # 对输出进行标准化

        x = self.relu0(x)  # 对输出使用ReLU激活函数处理

        x = self.pool0(x)  # 对输出进行池化操作
        x = self.denseblock1(x)  # 对输出进行第一个dense block的处理
        s4 = x  # 获取s4特征图
        x = self.transition1(x)  # 对输出进行第一个过渡层的处理
        x = self.denseblock2(x)  # 对输出进行第二个dense block的处理
        s8 = x  # 获取s8特征图
        x = self.transition2(x)  # 对输出进行第二个过渡层的处理
        x = self.denseblock3(x)  # 对输出进行第三个dense block的处理
        s16 = x  # 获取s16特征图     # torch.Size([2, 1792, 24, 24])
        return {'s4': s4, 's8': s8, 's16': s16}


# 匹配模块
class Matcher(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Conv(in_c, out_c, 1, 1, 0)  # 初始化一个卷积层
        self.short = nn.Parameter(torch.Tensor([1, 1]))  # 定义一个短时序列的参数
        self.long = nn.Parameter(torch.Tensor([-1, -1]))  # 定义一个长时序列的参数

    def get_key(self, x):  # 将输入的feature map通过卷积层和L2归一化得到每个pixel对应的向量表示
        x = self.conv(x)  # 使用上面定义好的卷积层处理输入x
        key = x / x.norm(dim=1, keepdim=True)  # 对x进行归一化，得到key
        return key

    @staticmethod
    def get_fine_temp(key, seg_16):  # 将输入的向量表示与二元分割图中的背景/前景掩码相乘，得到背景/前景的fine-grained templates。
        B, _, H, W = key.size()

        key = key.view(B, -1, H * W).transpose(1, 2)  # 将key的shape变为[B, H*W, C]，并转置
        # 计算背景和前景的fine-grained templates
        bg_temp = key * seg_16[:, 0].view(B, H * W, 1)
        fg_temp = key * seg_16[:, 1].view(B, H * W, 1)
        fine_temp = [bg_temp, fg_temp]
        return fine_temp

    @staticmethod
    def get_coarse_temp(fine_temp):  # 将fine-grained templates对所有像素点的向量表示进行求和并做L2归一化，得到背景/前景的coarse-grained templates
        # 计算背景和前景的coarse-grained templates
        bg_key_sum = torch.sum(fine_temp[0], dim=1, keepdim=True).clamp(min=1e-7)  # 对背景临时模板的所有值进行求和
        fg_key_sum = torch.sum(fine_temp[1], dim=1, keepdim=True).clamp(min=1e-7)  # 对前景临时模板的所有值进行求和
        bg_temp = bg_key_sum / bg_key_sum.norm(dim=2, keepdim=True)  # 对背景模板进行归一化
        fg_temp = fg_key_sum / fg_key_sum.norm(dim=2, keepdim=True)  # 对前景模板进行归一化
        coarse_temp = [bg_temp, fg_temp]  # 将背景模板和前景模板返回
        return coarse_temp

    @staticmethod
    def forward(key, sds, state):
        B, _, H, W = key.size()  # 获取输入key的batch size、通道数、高和宽
        key = key.view(B, -1, H * W)  # 将输入key reshape成(B, HW, feature_dim)
        sds = sds.view(B, H * W, H, W)  # 将输入sds reshape成(B, HW, H, W)

        # 全局匹配
        score = torch.bmm(state['global'][0], key).view(B, H * W, H, W)  # 计算global branch的background分数
        bg_score = torch.max(score, dim=1, keepdim=True)[0]  # 取最大值，得到每个像素位置的background最大分数
        score = torch.bmm(state['global'][1], key).view(B, H * W, H, W)  # 计算global branch的foreground分数
        fg_score = torch.max(score, dim=1, keepdim=True)[0]  # 取最大值，得到每个像素位置的foreground最大分数
        global_score = torch.cat([bg_score, fg_score], dim=1)  # 将background和foreground分数拼接起来

        # 局部匹配
        score = torch.bmm(state['local'][0], key).view(B, H * W, H, W) * sds  # 计算local branch的background分数
        bg_score = torch.max(score, dim=1, keepdim=True)[0]  # 取最大值，得到每个像素位置的background最大分数
        score = torch.bmm(state['local'][1], key).view(B, H * W, H, W) * sds  # 计算local branch的foreground分数
        fg_score = torch.max(score, dim=1, keepdim=True)[0]  # 取最大值，得到每个像素位置的foreground最大分数
        local_score = torch.cat([bg_score, fg_score], dim=1)  # 将background和foreground分数拼接起来
        fine_score = torch.cat([global_score, local_score], dim=1)  # 将global和local的分数拼接起来

        # 整体匹配
        bg_score = torch.bmm(state['overall'][0], key).view(B, 1, H, W)  # 计算overall branch的background分数
        fg_score = torch.bmm(state['overall'][1], key).view(B, 1, H, W)  # 计算overall branch的foreground分数
        overall_score = torch.cat([bg_score, fg_score], dim=1)  # 将background和foreground分数拼接起来

        # 短期匹配
        bg_score = torch.bmm(state['short'][0], key).view(B, 1, H, W)
        fg_score = torch.bmm(state['short'][1], key).view(B, 1, H, W)
        short_score = torch.cat([bg_score, fg_score], dim=1)

        # 长期匹配
        bg_score = torch.bmm(state['long'][0], key).view(B, 1, H, W)
        fg_score = torch.bmm(state['long'][1], key).view(B, 1, H, W)
        long_score = torch.cat([bg_score, fg_score], dim=1)
        coarse_score = torch.cat([overall_score, short_score, long_score], dim=1)

        # 聚合匹配分数
        matching_score = torch.cat([fine_score, coarse_score], dim=1)
        return matching_score


# 解码模块
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvRelu(1792, 256, 1, 1, 0)  # 定义第一层卷积，输入通道数为1024，输出通道数为256，卷积核为1x1，步长为1，填充为0，激活函数为ReLU
        # 定义第一次特征融合，将conv1的输出、匹配分数和上一帧的16倍下采样结果拼接起来，输入通道数为256+10+2，输出通道数为256，卷积核为3x3，步长为1，填充为1，激活函数为ReLU
        self.blend1 = ConvRelu(256 + 10 + 2, 256, 3, 1, 1)
        self.cbam1 = CBAM(256)  # 定义CBAM模块1
        self.deconv1 = DeConv(256, 2, 4, 2, 1)  # 定义第一层反卷积，输入通道数为256，输出通道数为2，卷积核为4x4，步长为2，填充为1

        self.conv2 = ConvRelu(512, 256, 1, 1, 0)  # 定义第二层卷积，输入通道数为512，输出通道数为256，卷积核为1x1，步长为1，填充为0，激活函数为ReLU
        # 定义第二次特征融合，将conv2的输出和上一次反卷积结果拼接起来，输入通道数为256+2，输出通道数为256，卷积核为3x3，步长为1，填充为1，激活函数为ReLU
        self.blend2 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam2 = CBAM(256)  # 定义CBAM模块2
        self.deconv2 = DeConv(256, 2, 4, 2, 1)  # 定义第二层反卷积，输入通道数为256，输出通道数为2，卷积核为4x4，步长为2，填充为1

        self.conv3 = ConvRelu(256, 256, 1, 1, 0)  # 定义第三层卷积，输入通道数为256，输出通道数为256，卷积核为1x1，步长为1，填充为0，激活函数为ReLU
        # 定义第二次特征融合，将conv3的输出和上一次反卷积结果拼接起来，输入通道数为256+2，输出通道数为256，卷积核为3x3，步长为1，填充为1，激活函数为ReLU
        self.blend3 = ConvRelu(256 + 2, 256, 3, 1, 1)
        self.cbam3 = CBAM(256)  # # 定义CBAM模块2
        self.deconv3 = DeConv(256, 2, 6, 4, 1)  # 定义第二层反卷积，输入通道数为256，输出通道数为2，卷积核为6x6，步长为4，填充为1

    def forward(self, feats, matching_score, prev_seg_16):
        x = torch.cat([self.conv1(feats['s16']), matching_score, prev_seg_16], dim=1)  # 将输入的特征、匹配分数与上一帧分割结果在通道维度上拼接起来
        s8 = self.deconv1(self.cbam1(self.blend1(x)))  # 经过第一个卷积层和CBAM模块进行特征融合，再通过反卷积进行上采样得到s8尺寸的特征

        x = torch.cat([self.conv2(feats['s8']), s8], dim=1)  # 在s8特征基础上进行第二次特征融合和上采样，得到s4尺寸的特征
        s4 = self.deconv2(self.cbam2(self.blend2(x)))

        x = torch.cat([self.conv3(feats['s4']), s4], dim=1)  # 在s4特征基础上进行第三次特征融合和上采样，最终得到输出的分割结果
        final_score = self.deconv3(self.cbam3(self.blend3(x)))
        return final_score


# VOS 模型
class VOS(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()  # 初始化一个Encoder实例
        self.matcher = Matcher(1792, 512)  # 初始化一个Matcher实例，输入通道数为1024，输出通道数为512
        self.decoder = Decoder()  # 初始化一个Decoder实例

    def get_init_state(self, key, given_seg):
        # 对给定分割结果进行平均池化，从而得到s16尺寸的上一帧分割结果prev_seg_16，并将其存入state中
        given_seg_16 = F.avg_pool2d(given_seg, 16)

        # 对prev_seg_16在通道维度上求和得到seg_sum，并将其存入state中
        seg_temp = torch.sum(given_seg_16, dim=[2, 3]).unsqueeze(2)

        # 调用Matcher实例的方法获取精细匹配模板fine_temp，并将其存入state中
        fine_temp = self.matcher.get_fine_temp(key, given_seg_16)

        # 调用Matcher实例的方法获取粗糙匹配模板coarse_temp，并将其存入state中
        coarse_temp = self.matcher.get_coarse_temp(fine_temp)

        # 初始化一个空字典state
        state = {'prev_seg_16': given_seg_16,
                 'seg_sum': seg_temp,
                 'global': fine_temp,
                 'local': fine_temp,
                 'overall': coarse_temp,
                 'short': coarse_temp,
                 'long': coarse_temp}

        # 返回初始化的状态state
        return state

    def update_state(self, key, pred_seg, state):  # 定义一个更新状态的函数，接收三个参数，分别是key、预测分割pred_seg和当前状态state

        # 更新prev seg和seg sum
        pred_seg_16 = F.avg_pool2d(pred_seg, 16)  # 对预测分割pred_seg进行2D平均池化，得到pred_seg_16
        state['prev_seg_16'] = pred_seg_16  # 将pred_seg_16保存到状态字典中
        seg_sum = torch.sum(pred_seg_16, dim=[2, 3]).unsqueeze(2)  # 求出pred_seg_16在第2和3维上的元素和，得到seg_sum，再增加一维，得到形状为(N, 1, 1)的张量
        state['seg_sum'] = state['seg_sum'] + seg_sum  # 将seg_sum加到状态字典中保存的seg_sum上

        # 更新local matching template
        fine_temp = self.matcher.get_fine_temp(key, pred_seg_16)  # 使用匹配器matcher获取key对应的细节模板fine_temp
        state['local'] = fine_temp  # 将fine_temp保存到状态字典中

        # 更新overall matching template
        coarse_temp = self.matcher.get_coarse_temp(fine_temp)  # 使用matcher获取粗略模板coarse_temp
        dynamic_inertia = 1 - seg_sum / state['seg_sum']  # 计算动态惯性dynamic_inertia
        bg_temp = dynamic_inertia[:, :1] * state['overall'][0] + (1 - dynamic_inertia[:, :1]) * coarse_temp[0]  # 计算背景模板bg_temp
        fg_temp = dynamic_inertia[:, 1:] * state['overall'][1] + (1 - dynamic_inertia[:, 1:]) * coarse_temp[1]  # 计算前景模板fg_temp
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)  # 对bg_temp进行2范数归一化，得到形状为(N, 1, L)的张量
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)  # 对fg_temp进行2范数归一化，得到形状为(N, K, L)的张量
        state['overall'] = [bg_temp, fg_temp]  # 将bg_temp和fg_temp保存到状态字典中作为overall matching template

        # 更新短期匹配模板
        short_inertia = 1 / (1 + torch.exp(self.matcher.short))  # 计算短期惯性
        bg_temp = short_inertia[0] * state['short'][0] + (1 - short_inertia[0]) * coarse_temp[0]  # 背景模板计算
        fg_temp = short_inertia[1] * state['short'][1] + (1 - short_inertia[1]) * coarse_temp[1]  # 前景模板计算
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)  # 对背景模板进行归一化处理
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)  # 对前景模板进行归一化处理
        state['short'] = [bg_temp, fg_temp]  # 更新状态的短期变量

        # 更新长期匹配模板
        long_inertia = 1 / (1 + torch.exp(self.matcher.long))  # 计算长期惯性
        bg_temp = long_inertia[0] * state['long'][0] + (1 - long_inertia[0]) * coarse_temp[0]  # 背景模板计算
        fg_temp = long_inertia[1] * state['long'][1] + (1 - long_inertia[1]) * coarse_temp[1]  # 前景模板计算
        bg_temp = bg_temp / bg_temp.norm(dim=2, keepdim=True)  # 对背景模板进行归一化处理
        fg_temp = fg_temp / fg_temp.norm(dim=2, keepdim=True)  # 对前景模板进行归一化处理
        state['long'] = [bg_temp, fg_temp]  # 更新状态的长期变量
        return state  # 返回更新后的状态

    def forward(self, feats, key, sds, state):
        matching_score = self.matcher(key, sds, state)  # 使用关键帧和区域提议计算匹配得分
        final_score = self.decoder(feats, matching_score, state['prev_seg_16'])  # 使用输入特征、匹配得分和上一个 16 帧的分割结果计算最终得分
        return final_score  # 返回最终得分


# TBD 模型

class STAVOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.vos = VOS()
        self.dsf = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid())  # 计算空间距离分数

    def forward(self, imgs, given_masks, dist, val_frame_ids=None, sta_id=0, mode="train"):
        print("sta_id:", sta_id)
        B, L, _, H, W = imgs.size()  # 获取视频帧数、高和宽
        padding = get_padding(H, W, 16)  # 计算需要添加的填充大小
        if tuple(padding) != (0, 0, 0, 0):
            imgs, given_masks = attach_padding(imgs, given_masks, padding)  # 对图像和掩码进行填充操作
        # 计算空间距离得分
        sds = self.dsf(dist.view(-1, 1))

        object_ids = given_masks[0].unique().tolist()  # 获取第一帧中出现的物体ID列表    [0,1]
        object_ids.remove(0)  # 如果有0在其中，则删除     # [1]
        # print("object_ids:", object_ids)
        # if object_ids is None:
        #     break

        score_lst = []
        mask_lst = [given_masks[0]]  # 初始化掩码列表为sta帧的掩码

        # 最佳帧###############################################################################################
        # 提取特征
        with torch.no_grad():
            sta_feats = self.vos.encoder(imgs[:, sta_id])  # 获取帧序列中sta帧的特征
        sta_key = self.vos.matcher.get_key(sta_feats['s16'])  # 获取sta特征对应的sta_key

        # 为最佳帧的每个对象创建状态字典
        sta_state = {}  # 初始化状态字典，键为对象id,值为状态
        for k in object_ids:
            given_seg = torch.cat([given_masks[0] != k, given_masks[0] == k], dim=1).float()  # 将最佳帧掩码转化为二元掩码
            sta_state[k] = self.vos.get_init_state(sta_key, given_seg)  # 获取初始状态



        # 前续帧###############################################################################################
        state = sta_state.copy()
        for i in range(sta_id - 1, -1, -1):
            # 提取特征
            with torch.no_grad():
                feats = self.vos.encoder(imgs[:, i])  # 获取帧序列中第i帧的特征
            key = self.vos.matcher.get_key(feats['s16'])  # 获取特征对应的key

            # 查询帧预测
            pred_seg = {}
            for k in object_ids:
                final_score = self.vos(feats, key, sds, state[k])  # 解码
                pred_seg[k] = torch.softmax(final_score, dim=1) # final_score torch.Size([2, 2, 384, 384])
                score_lst.append(final_score)   # pred_seg[k] torch.Size([2, 2, 384, 384])

            # 聚合对象
            if mode != "train":
                pred_mask, pred_seg = aggregate_objects(pred_seg, object_ids)
                if given_masks[i] is not None:
                    pred_mask[given_masks[i] != 0] = 0
                    mask_lst.append(pred_mask + given_masks[i])
                else:
                    if val_frame_ids is not None:
                        if val_frame_ids[0] + i in val_frame_ids:
                            mask_lst.append(pred_mask)
                    else:
                        mask_lst.append(pred_mask)
            # 更新状态
            if i != 0:
                for k in object_ids:
                    state[k] = self.vos.update_state(key, pred_seg[k], state[k])

        mask_lst = mask_lst[::-1]
        score_lst = score_lst[::-1]




        # 后续帧###############################################################################################
        state = sta_state.copy()
        for i in range(sta_id + 1, L):
            # 提取特征
            with torch.no_grad():
                feats = self.vos.encoder(imgs[:, i])  # 获取帧序列中第i帧的特征
            key = self.vos.matcher.get_key(feats['s16'])  # 获取特征对应的key

            # 查询帧预测
            pred_seg = {}
            for k in object_ids:
                final_score = self.vos(feats, key, sds, state[k])  # 解码
                pred_seg[k] = torch.softmax(final_score, dim=1) # final_score torch.Size([2, 2, 384, 384])
                score_lst.append(final_score)   # pred_seg[k] torch.Size([2, 2, 384, 384])

            # 聚合对象
            if mode != "train":
                pred_mask, pred_seg = aggregate_objects(pred_seg, object_ids)
                if given_masks[i] is not None:
                    pred_mask[given_masks[i] != 0] = 0
                    mask_lst.append(pred_mask + given_masks[i])
                else:
                    if val_frame_ids is not None:
                        if val_frame_ids[0] + i in val_frame_ids:
                            mask_lst.append(pred_mask)
                    else:
                        mask_lst.append(pred_mask)
            # 更新状态
            if i != L - 1:
                for k in object_ids:
                    state[k] = self.vos.update_state(key, pred_seg[k], state[k])

        # 生成输出
        output = {}
        # print(imgs.size())                              # torch.Size([2, 10, 3, 384, 384])
        # print("sta_id:", sta_id)
        if mode == "train":
            # print(len(score_lst), score_lst[0].size())  # 9 torch.Size([2, 2, 384, 384])
            output['scores'] = torch.stack(score_lst, dim=1)
            output['scores'] = detach_padding(output['scores'], padding)
        else:
            # print(len(mask_lst), mask_lst[0].size())        # 1 torch.Size([2, 1, 384, 384])
            output['masks'] = torch.stack(mask_lst, dim=1)
            output['masks'] = detach_padding(output['masks'], padding)
        return output

