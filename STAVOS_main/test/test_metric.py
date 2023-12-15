# -*- coding:utf-8 -*-
import json
import os
import cv2
from sklearn.metrics import confusion_matrix


def get_metric(predict_mask, true_mask):  # predict和mask都为0，1二值numpy
    tn, fp, fn, tp = confusion_matrix(true_mask.flatten(), predict_mask.flatten()).ravel()
    Recall = tp / (tp + fn)
    Precision = tp / (tp + fp)
    Accuracy = (tp + tn) / (tp + fp + fn + tn)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    IoU = tp / (tp + fp + fn)
    MIoU = (tp / (tp + fp + fn) + tn / (tn + fn + fp)) / 2

    return Recall, Precision, Accuracy, F1, IoU, MIoU


def get_metric_json(predict_path, true_path, save_path):
    video_names = os.listdir(predict_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    video_recall = {}
    video_precision = {}
    video_accuracy = {}
    video_f1 = {}
    video_iou = {}
    video_miou = {}

    for video_name in video_names:
        print(video_name)

        predict_names = os.listdir(predict_path + video_name)
        true_names = os.listdir(true_path + video_name)

        recall_list = []
        precision_list = []
        accuracy_list = []
        f1_list = []
        iou_list = []
        miou_list = []
        i = 0
        for predict_name, true_name in zip(predict_names, true_names):

            predict_mask = cv2.imread(predict_path + video_name + "/" + predict_name)
            true_mask = cv2.imread(true_path + video_name + "/" + true_name)
            recall, precision, accuracy, f1, iou, miou = get_metric(predict_mask, true_mask)
            print(i, "recall:", recall, "precision:", precision, "accuracy:", accuracy, "f1:", f1, "iou:", iou, "miou:", miou)
            recall_list.append(recall)
            precision_list.append(precision)
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            iou_list.append(iou)
            miou_list.append(miou)
            i += 1


        video_recall[video_name] = sum(recall_list) / len(recall_list)
        video_precision[video_name] = sum(precision_list) / len(precision_list)
        video_accuracy[video_name] = sum(accuracy_list) / len(accuracy_list)
        video_f1[video_name] = sum(f1_list) / len(f1_list)
        video_iou[video_name] = sum(iou_list) / len(iou_list)
        video_miou[video_name] = sum(miou_list) / len(miou_list)
        print(video_name, "平均recall:", video_recall[video_name], "平均precision:", video_precision[video_name], "平均accuracy:", video_accuracy[video_name], "平均f1:", video_f1[video_name], "平均iou:", video_iou[video_name], "平均miou:", video_miou[video_name])

        video_metric = {"recall_list": recall_list, "video_recall": video_recall[video_name],
                        "precision_list": precision_list, "video_precision": video_precision[video_name],
                        "accuracy_list": accuracy_list, "video_accuracy": video_accuracy[video_name],
                        "f1_list": f1_list, "video_f1": video_f1[video_name],
                        "iou_list": iou_list, "video_iou": video_iou[video_name],
                        "miou_list": miou_list, "video_miou": video_miou[video_name]}

        with open(save_path + video_name + ".json", "w") as json_file:
            # 将json数据写入json文件
            json.dump(video_metric, json_file)

    dataset_recall = sum(video_recall[video_name] for video_name in video_recall) / len(video_recall)
    dataset_precision = sum(video_precision[video_name] for video_name in video_precision) / len(video_precision)
    dataset_accuracy = sum(video_accuracy[video_name] for video_name in video_accuracy) / len(video_accuracy)
    dataset_f1 = sum(video_f1[video_name] for video_name in video_f1) / len(video_f1)
    dataset_iou = sum(video_iou[video_name] for video_name in video_iou) / len(video_iou)
    dataset_miou = sum(video_miou[video_name] for video_name in video_iou) / len(video_miou)

    dataset_metric = {"video_recall": video_iou, "dataset_recall": dataset_recall,
                      "video_precision": video_iou, "dataset_precision": dataset_precision,
                      "video_accuracy": video_iou, "dataset_accuracy": dataset_accuracy,
                      "video_f1": video_iou, "dataset_f1": dataset_f1,
                      "video_iou": video_iou, "dataset_iou": dataset_iou,
                      "video_miou": video_iou, "dataset_miou": dataset_miou, }
    with open(save_path + "total.json", "w") as json_file:
        json.dump(dataset_metric, json_file)


if __name__ == "__main__":
    predict_path = "../../test_predict/N_better/"
    true_path = "../../Data-medaka-Lateral-right/better/Annotations/480p/"
    save_path = "../../test_predict/metric/"

    get_metric_json(predict_path, true_path, save_path)
