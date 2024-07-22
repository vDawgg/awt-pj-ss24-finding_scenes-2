import difflib
import json
import os

import pandas as pd
from pycocotools.coco import COCO
from eval_scripts.eval import COCOEvalCap

def make_json_from_csvs():
    ds = {
        "images": [],
        "annotations": [],
    }
    counter = 0
    ds_ids = []
    for csv in os.listdir('./dataset/test'):
        print(csv)
        df = pd.read_csv(os.path.join('./dataset', 'test', csv))

        img_ids = df['Source Filename'].tolist()
        ds_ids.extend(img_ids)
        for id in img_ids:
            if {"id": id} not in ds["images"]:
                ds['images'].append({
                    "id": id,
                })

        captions = df['CAPTION'].tolist()
        for id, cap in zip(img_ids, captions):
            ds['annotations'].append({
                "image_id": id,
                "id": counter,
                "caption": cap,
            })
            counter += 1

    json.dump(ds, open('./dataset/ds.json', 'w'))

    preds = []
    pred_ids = []
    for csv in os.listdir('./dataset/pred'):
        df = pd.read_csv(os.path.join('./dataset', 'pred', csv))

        img_ids = df['Source Filename'].tolist()
        pred_ids.extend(img_ids)
        captions = df['CAPTION'].tolist()

        for id, cap in zip(img_ids, captions):
            preds.append({
                "image_id": id,
                "caption": cap,
            })

    json.dump(preds, open('./dataset/pred.json', 'w'))

    print(set(pred_ids).difference(set(ds_ids)))

def eval_predictions():
    annotations = './dataset/ds.json'
    results = './dataset/pred.json'

    coco = COCO(annotations)
    coco_result = coco.loadRes(results)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(metric, score)

if __name__ == '__main__':
    make_json_from_csvs()
    eval_predictions()
