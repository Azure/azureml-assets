`ocsort_yolox_x_crowdhuman_mot17-private-half` model is from <a href="https://github.com/open-mmlab/mmtracking/tree/v0.14.0" target="_blank">OpenMMLab's MMTracking library</a>. This model is <a href="https://github.com/open-mmlab/mmtracking/blob/master/configs/mot/ocsort/metafile.yml#L24" target="_blank">reported</a> to obtain MOTA: 77.8, IDF1: 78.4 for video-multi-object-tracking task on <a href="https://motchallenge.net/data/MOT17/" target="_blank">MOT17-half-eval dataset</a>.

Multi-Object Tracking (MOT) has rapidly progressed with the development of object detection and re-identification. However, motion modeling, which facilitates object association by forecasting short-term trajec- tories with past observations, has been relatively under-explored in recent years. Current motion models in MOT typically assume that the object motion is linear in a small time window and needs continuous observations, so these methods are sensitive to occlusions and non-linear motion and require high frame-rate videos. In this work, we show that a simple motion model can obtain state-of-the-art tracking performance without other cues like appearance. We emphasize the role of “observation” when recovering tracks from being lost and reducing the error accumulated by linear motion models during the lost period. We thus name the proposed method as Observation-Centric SORT, OC-SORT for short. It remains simple, online, and real-time but improves robustness over occlusion and non-linear motion. It achieves 63.2 and 62.1 HOTA on MOT17 and MOT20, respectively, surpassing all published methods. It also sets new states of the art on KITTI Pedestrian Tracking and DanceTrack where the object motion is highly non-linear.

> The above abstract is from MMTracking website. Review the <a href="https://github.com/open-mmlab/mmtracking/tree/v0.14.0/configs/mot/ocsort" target="_blank">original-model-card</a> to understand the data used to train the model, evaluation metrics, license, intended uses, limitations and bias before using the model.
### Inference samples

Inference type|Python sample (Notebook)|CLI with YAML
|--|--|--|
Real time|<a href="https://aka.ms/azureml-video-mutli-object-tracking-online-inference" target="_blank">video-multi-object-tracking-online-endpoint.ipynb</a>|<a href="https://aka.ms/cli-video-multi-object-tracking-online-inference" target="_blank">video-multi-object-tracking-online-endpoint.sh</a>|

### Finetuning samples

Task|Use case|Dataset|Python sample (Notebook)|CLI with YAML
|---|--|--|--|--|
Video multi-object tracking|Video multi-object tracking|[MOT17 tiny](https://download.openmmlab.com/mmtracking/data/MOT17_tiny.zip)|<a href="https://aka.ms/azureml-video-multi-object-tracking-finetune" target="_blank">mot17-tiny-video-multi-object-tracking.ipynb</a>|<a href="https://aka.ms/cli-video-multi-object-tracking-finetune" target="_blank">mot17-tiny-video-multi-object-tracking.sh</a>|


### Sample inputs and outputs (for real-time inference)

#### Sample input

```json
{
  "input_data": {
    "columns": [
      "video"
    ],
    "data": ["video_link"]
  }
}
```

Note: "video_link" should be a publicly accessible url.

#### Sample output

```json
[
  {
    "det_bboxes": [
      {
        "box": {
          "topX": 703.9149780273,
          "topY": -5.5951070786,
          "bottomX": 756.9875488281,
          "bottomY": 158.1963806152
        },
        "label": 0,
        "score": 0.9597821236
      },
      {
        "box": {
          "topX": 1487.9072265625,
          "topY": 67.9468841553,
          "bottomX": 1541.1591796875,
          "bottomY": 217.5476837158
        },
        "label": 0,
        "score": 0.9568068385
      }
    ],
    "track_bboxes": [
      {
        "box": {
          "instance_id": 0,
          "topX": 703.9149780273,
          "topY": -5.5951070786,
          "bottomX": 756.9875488281,
          "bottomY": 158.1963806152
        },
        "label": 0,
        "score": 0.9597821236
      },
      {
        "box": {
          "instance_id": 1,
          "topX": 1487.9072265625,
          "topY": 67.9468841553,
          "bottomX": 1541.1591796875,
          "bottomY": 217.5476837158
        },
        "label": 0,
        "score": 0.9568068385
      }
    ],
    "frame_id": 0,
    "video_url": "video_link"
  }
]
```

