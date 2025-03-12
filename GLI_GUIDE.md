# :computer: CLI Demo

Before diving into the command lines, we introduce `Task` to bucket different usage cases depending on the data constraints in input and output domains (e.g., if the ordering is available).

### Task
| Task | Type of NVS | Format of `<data_path>` | Target Views Sorted? | Input and Target Views Sorted? | Recommended Usage |
|:--------:|:--------:|:--------:|:--------:|:--------:| :--------:| 
| `img2img`   | set NVS     | folder (parsable by `ReconfusionParser`)     | :x:     | :x:     | evaluation, benchmarking |
| `img2vid`    | trajectory NVS     | folder (parsable by `ReconfusionParser`)     | :white_check_mark:     | :white_check_mark:     | evaluation, benchmarking
| `img2trajvid_s-prob`    | trajectory NVS     | single image     | :white_check_mark:     | :white_check_mark:     | general |
| `img2trajvid`    | trajectory NVS     | folder (parsable by `ReconfusionParser`)     | :white_check_mark:     | :x:    | general |

#### Format of `<data_path>`

For `img2trajvid_s-prob` task, we are generating a trajectory video following preset camera motions or effects given only one input image, the data format as simple as

```
<data_path>/
  ├── scene_1.png
  ├── scene_2.png
  └── scene_3.png
```

For all the other tasks, we use a folder for each scene that is parsable by `ReconfusionParser` (see `seva/data_io.py`). It contains (1) a subdirectory containing all views; (2) `transforms.json` defining the intrinsics and extrinsics for each image; and (3) `train_test_split_{P}.json` file splitting the input and target views, with `P` indicating the number of the input views.

Target views is available if you the data are from academic sources, but in the case where target views is unavailble, we will create dummy black images as placeholders.

We provide in `demo` folder hosted <a href="">here</a> two examplar scenes for you to take reference from. The general data structure follows
```
<data_path>/
├── scene_1/
    ├── train_test_split_1.json # for single-view regime
    ├── train_test_split_6.json # for sparse-veiw regime
    ├── train_test_split_32.json # for semi-dense-view regime
    ├── transforms.json
    └── images/
        ├── image_0.png
        ├── image_1.png
        ├── ...
        └── image_1000.png
├── scene_2
└── scene_3
```

#### Recommended Usage
- `img2img` and `img2vid` are recommended to be used for evaluation and benchmarking. These two tasks are used for the quantitative evalution throught the <a href="https://arxiv.org/abs/0000.0000">paper</a>. The data is converted from academic datasets so the groundtruth target views are available for metric computation. `img2vid` requries both the input and target views to be sorted, which is usually not guaranteed in general usage.
- `img2trajvid_s-prob` is for general usage but only for single-view regime and fixed preset camera control.
- `img2trajvid` is the task designed for general usage since it does not need the ordering of the input views. This is the task used in the gradio demo.


### `img2img`

```
python demo.py \
    --task img2img \
    --num_inputs 3 \ 
    --dataset_path /weka/home-hangg/datasets/reconfusion/dl3dv10 \
    --video_save_fps 10
```
- Preprocessing of images will perform center cropping. All input and output are square images by default.
- The above command works for the dataset without trajectory prior (e.g., DL3DV-140). When the trajectory prior is available given a benchmarking dataset, for example, `orbit` trajectory prior for the CO3D dataset, we use the `nearest-gt` chunking strategy by setting `--use_traj_prior True --traj_prior orbit --chunking_strategy nearest-gt --save_subdir  2pass_chunk-nearest-gt`. We find this leads to more 3D consistent results.
- For all the single-view conditioning test scenarios: use `--num_input_frames 1 --camera_scales 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0` to sweeping 20 different camera scales
- For the single-view conditioning test scenario on the RealEstate10K dataset, we find increasing cfg is helpful: we additionally set `--cfg 6.0` (cfg is `2.0` by default).
- For the evaluation on the semi-dense-view regime (i.e., DL3DV-140 and Tanks and Temples dataset) with `32` input views, we zero-shot extend `T` to fit all input and target views in one forward. Specifically, we set `--T 90` for the DL3DV-140 dataset and `--T 80` for the Tanks and Temples dataset.
- For the evaluation on ViewCrafter split (including the RealEastate10K, CO3D, and Tanks and Temples dataset), we find zero-shot extending `T` to `25` to fit all input and target views in one forward is better. Also, the V split uses the original aspect ratio of the image: we therefore set `--T 25 --L_short 576`.


### `img2vid`

```
python demo.py \
    --task img2vid \
    --num_inputs 3 \
    --data_path /weka/home-hangg/datasets/reconfusion/dl3dv10 \
    --use_traj_prior True \
    --chunk_strategy interp \
    --save_subdir 2pass_chunk-interp \
    --replace_or_include_input True
```


- Preprocessing of images will perform center cropping. All input and output are square images by default.
- We use `interp` chunking strategy by default.
- For the evaluation on ViewCrafter split (including the RealEastate10K, CO3D, and Tanks and Temples dataset), we find zero-shot extending `T` to `25` to fit all input and target views in one forward is better. Also, the V split uses the original aspect ratio of the image: we therefore set `--T 25 --L_short 576`.

### `img2traj_s-prob`

```
python demo.py \
    --task img2trajvid_s-prob \
    --traj_prior orbit \
    --data_path /weka/home-jensen/scena-image/scene \
    --data_items 64d39070-8596-11ef-addc-5556603eb4c1.png \
     --cfg 4.0,2.0 \
     --guider 1,2 \
     --num_targets 111 \
     --L_short 576 \
     --use_traj_prior True \
     --chunk_strategy interp\
     --save_subdir 2pass_chunk-interp_cfg-4+2-gd-1+2_num-111

```

- Default cfg is dependent on `traj_prior`. 
- Default chunking strategy is `interp`. 
- Default guider is `--guider 1,2` (instead of `1`, `1` still works but `1,2` is slightly better).

### `img2trajvid`

#### Sparse-view regime ($P\leq 8$)

```
python demo.py \    
    --task img2trajvid \                       
    --data_path /weka/home-hangg/datasets/reconfusion/demo \
    --data_items telephone_booth \
    --cfg 3.0,2.0  \
    --use_traj_prior True \
    --chunk_strategy interp-gt \
    --save_subdir 2pass_chunk-interp-gt_cfg-3+2 
```

- Default cfg should be set to `3,2` (`3` being cfg for the first pass, and `2` being the cfg for the second pass). Try to increase the cfg for the first pass from `3` to higher values if you observe blurry areas (usually happens for harder scenes with a fair amount of unseen regions).
- Default chunking strategy should be set to `interp+gt` (instead of `interp`, `interp` can work but usually a bit worse).
- The `--chunk_strategy_first_pass` is default as `gt-nearest`. So it can automatically adapt when $P$ is large (up to a thousand frames).


#### Semi-dense-view regime ($P>9$)
```
python demo.py \
    --task img2trajvid \
    --num_inputs 32 \
    --data_path /weka/home-hangg/projects/stable-research/logs/input_data/ \
    --data_items 09c1414f1b_iphone_traj1 \
    --cfg 3.0  \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp \
    --save_subdir 2pass_chunk-interp_cfg-3
```
- Default cfg should be set to `3` (instead of `3,2`).
- Default chunking strategy should be set to `interp` (instead of `interp-gt`, `interp-gt` is also supported but the results look very bad).
- `T` will be `N,21` (X being extended `T` for the first pass, and `21` being the default `T` for the second pass). `N` is dynamically decided now. `N` can be manually overwritten in two ways. This is useful when you observe that there exist two very dissimilar adjacent anchors which make the interpolation in the second pass impossible.
    - `--T 96,21`: this overwrites the `T` in first pass to be exactly `96`.
    - `--num_prior_frames_ratio 1.2`: this enlarges T dynamically decided to be `1.2`x larger.

