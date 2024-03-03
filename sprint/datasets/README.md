# Dataset Generation

The dataset is already provided in the main README's download link, but this README and files here detail how we generate the dataset from ALFRED's original dataset.
Note these might not fully run right now after the refactoring as some huggingface dependencies have seemingly expired and I already provide the datset in the original README, but these scripts should give you an idea of how it was done.

If requested, I will work on fully refactoring these scripts to make them run and the commands fully working if you need it.

We modify the ALFRED dataset in a couple of ways: 

- We merge ALFRED's "GoTo" navigation skills with the following skill so that there are no commands regarding just navigating. We instead concatenate the skill actions/states together and use the instruction from the following skill.
- We augment the dataset (`llm_aggregate_dataset.py`) with new trajectories coming the LLM skill aggregation procedure specified in the paper Section III.B -- this increases the dataset size by ~2.5x
- We then preprocess (remove unnecessary "stop" actions for RL/imitation learning, and separate out our EVAL_SCENE, EVAL_INSTRUCT, and EVAL_LENGTH eval sets) and convert to our LMDB-based dataset for faster dataloading.

These scripts are mostly the same as ALFRED's dataset creation scripts with some minor modifications.

To generate the dataset from scratch, first download the alfred dataset by running 
```
cd sprint/alfred/data && sh download_data.sh json
```

Then, merge the GoTo skills:
```
python merge_goto.py --data_dir [ALFRED_DOWNLOAD_PATH] --save_dir [merged Goto dataset path]
```

Now, preprocess the trajectories
```
python preprocess.py ARGS
```

Extract resnet features, keep rerunning this script until there's no more error-ed out trajectories:
```
python extract_resnet.py --data [JSON_FOLDER] --gpu --visual_model resnet18 --filename feat_conv.pt --img_folder high_res_images --batch 1024 --skip_existing
```

Run the LLM dataset aggregation script (this is part of the SPRINT algorithm).
The 13b llama model we used is no longer available on huggingface, so to fully reproduce this you should follow the llama instructions to download LLaMA-13B (or just use Llama-2 or any other newer open source LLM). 
Right now this script defaults to 7b:
```
python llm_aggregate_dataset.py ARGS
```

Finally, convert them to the LMDB format:
```
python convert_data_to_lmdb.py ARGS
```