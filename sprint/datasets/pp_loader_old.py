import os
import json
import pprint
import torch
from tqdm import trange
import numpy as np
from sprint.alfred.gen.utils.image_util import decompress_mask
from sentence_transformers import SentenceTransformer
import collections
from torchvision import transforms
from PIL import Image

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# language_encode_model = SentenceTransformer('all-distilroberta-v1').to(device)


# data_path = 'data/full_2.1.0'
# train_data_path = os.path.join(data_path, "preprocess/train/")
# split_path = 'data/splits/oct21.json'
# vocab = torch.load(os.path.join("./data", "pp.vocab"))
# with open(split_path) as f:
#     splits = json.load(f)
#     pprint.pprint({k: len(v) for k, v in splits.items()})
# train = splits['train']
# train = train[0:100]
def process_images(image_path):
    files = os.listdir(image_path)
    fimages = sorted(
        [
            os.path.join(image_path, f)
            for f in files
            if (f.endswith(".png") or (f.endswith(".jpg")))
        ]
    )
    transform = transforms.Compose(
        [transforms.Resize((112, 112)), transforms.ToTensor(),]
    )
    image_loader = Image.open if isinstance(fimages[0], str) else Image.fromarray
    images = np.array([transform(image_loader(f)).numpy() for f in fimages])
    return images


def iterater(
    data,
    batch_size,
    data_path,
    set_type="train",
    folder_name="preprocess",
    num_threads=1,
    which_thread=1,
    load_mask=True,
):
    """
    breaks dataset into batch_size chunks for training
    """
    start = len(data) // num_threads * (which_thread - 1)
    end = len(data) // num_threads * which_thread
    for i in trange(start, end, batch_size, desc="batch"):
        tasks = data[i : i + batch_size]
        batch = [load_task_json(task, data_path, set_type, folder_name) for task in tasks]
        batch = [task for task in batch if task is not None]
        feat = []

        feat = featurize(batch, data_path, load_mask=load_mask, set_type=set_type)
        yield batch, feat


def load_task_json(task, data_path, set_type="train", folder_name="preprocess"):
    """
    load preprocessed json from disk
    """
    #     print(task['task'])
    json_path = os.path.join(
        data_path,
        folder_name, 
        set_type,
        task["task"],
        "ann_%d.json" % task["repeat_idx"],
    )
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return data
    else:
        print("json file not found: %s" % json_path)
        return None


def featurize(batch, data_path, load_mask=True, load_frames=True, set_type="train"):
    """
    tensorize and pad batch input
    """
    device = torch.device("cuda")
    feat = collections.defaultdict(list)

    for ex in batch:
        #########
        # inputs
        #########

        #         # serialize segments
        #         serialize_lang_action(ex)

        #         # goal and instr language
        lang_instr = ex["num"]["lang_instr"]

        #         encode_language = []

        #         for instruction in lang_instr:
        #             output = language_encode_model.encode(instruction)
        #             encode_language.append(output)
        #         feat['encode_language'].append(encode_language)
        feat["lang_goal_instr"].append(lang_instr)

        #         # load Resnet features from disk
        #         if load_frames and not self.test_mode:
        new_path = "/home/jesse/ALFRED_jiahui/data/json_2.1.0"
        root = os.path.join(new_path, set_type, *(ex["root"].split("/")[-2:]))
        #         print(root)

        #         image_path = []
        res_features_exists = os.path.exists(os.path.join(root, "feat_conv.pt"))

        if res_features_exists:
            if os.path.exists(os.path.join(root, "instance_masks")):
                feat["resnet_features"].append(
                    torch.load(os.path.join(root, "feat_conv.pt"))
                )
                if load_mask:
                    feat["instance_masks"].append(
                        process_images(os.path.join(root, "instance_masks"))
                    )

            else:
                feat["image_path"].append(os.path.join(root, "feat_conv.pt"))
                im = torch.load(os.path.join(root, "feat_conv.pt"))
                im = im[:-10]
                num_feat_frames = im.shape[0]
                feat["frames"].append(im)
        else:
            feat["rgb_frames"].append(0)
            #     process_images(os.path.join(root, "high_res_images"))
            # )

            # feat["resnet_features"].append(
            #    torch.load(os.path.join(root, "feat_conv.pt"))
            # )
            # feat["depth_path"].append(os.path.join(root, "depth_images"))
            if load_mask:
                feat["instance_masks"].append(
                    process_images(os.path.join(root, "instance_masks"))
                )

        #         im = im[:-10].to(device)

        start_action = -1
        action_switch_point = []

        for ii in range(len(ex["images"])):
            if ex["images"][ii]["low_idx"] == start_action + 1:
                start_action = start_action + 1
                action_switch_point.append(ii)

        start_skill = -1
        skill_switch_point = []
        for ii in range(len(ex["plan"]["low_actions"])):
            if ex["plan"]["low_actions"][ii]["high_idx"] == start_skill + 1:
                start_skill = start_skill + 1
                skill_switch_point.append(ii)

        feat["skill_switch_point"].append(skill_switch_point)
        feat["root"].append(root)
        feat["action_switch_point"].append(action_switch_point)
        action_low = []
        action_low_mask = []
        object_id_low_mask = []
        action_low_valid_interact = []
        for aa in ex["num"]["action_low"]:
            for a in aa:
                action_low.append(a["action"])
                action_low_valid_interact.append(a["valid_interact"])
                if a["mask"] == None:
                    action_low_mask.append(None)
                    object_id_low_mask.append(None)
                else:

                    #                     mask = decompress_mask(a['mask'])
                    mask = a["mask"]
                    object_id = a["object_id"]

                    #                     mask = torch.tensor(mask, device=device, dtype=torch.float)
                    action_low_mask.append(mask)
                    object_id_low_mask.append(object_id)

        feat["rewards"].append(ex["task"]["rewards"])
        feat["reward_upper_bound"].append(ex["task"]["reward_upper_bound"])
        feat["action_low"].append(action_low)
        feat["action_low_mask"].append(action_low_mask)
        feat["action_low_valid_interact"].append(action_low_valid_interact)
        feat["object_ids"].append(object_id_low_mask)

    return feat


def use_decompress_mask(compressed_mask):
    """
    decompress mask from json files
    """
    mask = np.array(decompress_mask(compressed_mask))

    mask = np.expand_dims(mask, axis=0)
    return mask
