
import os, argparse, torch, clip, pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.stats import wasserstein_distance

device  = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_embeddings_from_folder(folder_path: str):
    """Return a list[Tensor] of CLIP embeddings for every image in folder_path."""
    embeds = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isdir(fpath):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            emb = model.encode_image(preprocess(img).unsqueeze(0).to(device))
            embeds.append(emb.squeeze(0))        # (512,) fp16 on GPU
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
    if not embeds:
        raise ValueError(f"No valid images in {folder_path}")
    return embeds

def cos_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-4):
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)

def avg_pairwise(gen_embeds, train_embeds, metric_fn):
    vals = []
    for g in gen_embeds:
        for t in train_embeds:
            vals.append(metric_fn(g, t).item())
    return sum(vals) / len(vals)

def avg_wasserstein(a: torch.Tensor, b: torch.Tensor):
    return wasserstein_distance(a.cpu().detach().numpy(), b.cpu().detach().numpy())


def main(args):
    im_emb = {
        'dreambooth' : {},
        'instant_id' : {},
        'gpt_4o'     : {},
        'train'      : {},
    }
    categories = []

    for folder in tqdm(os.listdir(args.dreambooth_generated_images_folder), desc="DreamBooth"):
        categories.append(folder)
        im_emb['dreambooth'][folder] = clip_embeddings_from_folder(
            os.path.join(args.dreambooth_generated_images_folder, folder))

    for folder in tqdm(os.listdir(args.instant_id_generated_images_folder), desc="InstantID"):
        im_emb['instant_id'][folder] = clip_embeddings_from_folder(
            os.path.join(args.instant_id_generated_images_folder, folder))

    for folder in tqdm(os.listdir(args.gpt_4o_generated_paths), desc="GPT-4o"):
        im_emb['gpt_4o'][folder] = clip_embeddings_from_folder(
            os.path.join(args.gpt_4o_generated_paths, folder))

    for folder in tqdm(os.listdir(args.training_dataset_path), desc="Training set"):
        im_emb['train'][folder] = clip_embeddings_from_folder(
            os.path.join(args.training_dataset_path, folder))

    sim_report = {
        'category': [],
        'sim_dreambooth_train': [],
        'sim_instantid_train' : [],
        'sim_gpt4o_train'     : [],
    }
    wd_report  = {
        'category': [],
        'wd_dreambooth_train': [],
        'wd_instantid_train' : [],
        'wd_gpt4o_train'     : [],
    }

    for cat in tqdm(categories, desc="Evaluating"):
        train_embeds = im_emb['train'][cat]

        sim_report['category'].append(cat)
        wd_report ['category'].append(cat)

        sim_report['sim_dreambooth_train'].append(
            avg_pairwise(im_emb['dreambooth'][cat], train_embeds, cos_sim))
        sim_report['sim_instantid_train'].append(
            avg_pairwise(im_emb['instant_id'][cat],  train_embeds, cos_sim))
        sim_report['sim_gpt4o_train'].append(
            avg_pairwise(im_emb['gpt_4o'][cat],      train_embeds, cos_sim))

        wd_report['wd_dreambooth_train'].append(
            avg_pairwise(im_emb['dreambooth'][cat], train_embeds, avg_wasserstein))
        wd_report['wd_instantid_train'].append(
            avg_pairwise(im_emb['instant_id'][cat], train_embeds, avg_wasserstein))
        wd_report['wd_gpt4o_train'].append(
            avg_pairwise(im_emb['gpt_4o'][cat],     train_embeds, avg_wasserstein))

    pd.DataFrame(sim_report).to_csv(args.evaluation_report_path, index=False)
    pd.DataFrame(wd_report ).to_csv(
        args.evaluation_report_path.replace('.csv', '_wasserstein.csv'), index=False)
    
    pd.DataFrame(sim_report)[['sim_dreambooth_train', 'sim_instantid_train', 'sim_gpt4o_train']].mean().to_csv(
        args.evaluation_report_path.replace('.csv', '_means.csv' ), index=False)

    pd.DataFrame(wd_report)[['wd_dreambooth_train', 'wd_instantid_train', 'wd_gpt4o_train']].mean().to_csv(args.evaluation_report_path.replace('.csv', '_wasserstein_means.csv'), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dreambooth_generated_images_folder', default='/home/jupyter/genai_project/output_images/dreambooth')
    p.add_argument('--instant_id_generated_images_folder',  default='/home/jupyter/genai_project/output_images/instant_id')
    p.add_argument('--gpt_4o_generated_paths',              default='/home/jupyter/genai_project/output_images/gpt_4o')
    p.add_argument('--training_dataset_path',               default='/home/jupyter/genai_project/train_dataset_dreambooth')
    p.add_argument('--evaluation_report_path',              default='clip_pairwise_report.csv')
    args = p.parse_args()
    main(args)
