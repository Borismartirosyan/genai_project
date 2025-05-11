import pandas as pd
import argparse
import os
from PIL import Image
import clip
import torch
from tqdm import tqdm
from scipy.stats import wasserstein_distance


model, preprocess = clip.load("ViT-B/32", device='cuda')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def clip_mean_embedding(person_image_folder):
    images = []
    for image_path in os.listdir(person_image_folder):
        full_img_path = os.path.join(person_image_folder, image_path)
        if os.path.isdir(full_img_path):
            continue  # Skip subdirectories
        try:
            img = Image.open(full_img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Skipping file {full_img_path}. Reason: {e}")

    if not images:
        raise ValueError(f"No valid images found in {person_image_folder}")

    image_embeddings = [model.encode_image(preprocess(img).unsqueeze(0).to(device)) for img in images]
    image_embeddings = torch.stack(image_embeddings).squeeze(1)
    mean_image_embedding = torch.mean(image_embeddings, dim=0)

    return mean_image_embedding


def calculate_cos_similarity(a, b, epsilon=torch.tensor(0.0001, dtype=torch.float)):
    dot = a @ b
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)

    return (dot)/((norm_a*norm_b)+epsilon) 

def calculate_wasserstein_distance(a, b):
    a_cpu = a.detach().cpu().numpy()
    b_cpu = b.detach().cpu().numpy()
    return wasserstein_distance(a_cpu, b_cpu)

def main(args):

    image_embeddings = {}

    if os.path.exists(args.dreambooth_generated_images_folder):
        print('Evaluating Dreambooth Generated Images too')
        image_embeddings['dreambooth_images'] = {}
    if os.path.exists(args.instant_id_generated_images_folder):
        print('Evaluating InstantID Generated Images too')
        image_embeddings['instant_id_images'] = {}
    if os.path.exists(args.gpt_4o_generated_paths):
        print('Evaluating gpt-4o Generated Images too')
        image_embeddings['gpt_4o_images'] = {}
    
    if not os.path.exists(args.training_dataset_path):
        raise TypeError('The training data folder wasn`t found, please validate or provide the path correctly!')
    else:
        image_embeddings['training_images'] = {}
    
    categories = []
    for folder in tqdm(os.listdir(args.dreambooth_generated_images_folder)):
        image_embeddings['dreambooth_images'][f'{folder}'] = clip_mean_embedding(args.dreambooth_generated_images_folder+'/'+folder)
        categories.append(folder)

    for folder in tqdm(os.listdir(args.instant_id_generated_images_folder)):
        image_embeddings['instant_id_images'][f'{folder}'] = clip_mean_embedding(args.instant_id_generated_images_folder+'/'+folder)   
    
    for folder in tqdm(os.listdir(args.gpt_4o_generated_paths)):
        image_embeddings['gpt_4o_images'][f'{folder}'] = clip_mean_embedding(args.gpt_4o_generated_paths+'/'+folder)
    
    for folder in tqdm(os.listdir(args.training_dataset_path)):
        image_embeddings['training_images'][f'{folder}'] = clip_mean_embedding(args.training_dataset_path+'/'+folder)

    results = {
        'category' : categories,
        'sim_dreambooth_training' : [],
        'sim_instantid_training' : [],
        'sim_gpt_training' : [],
    }
    wasserstein_results = {
        'category' : categories,
        'wd_dreambooth_training' : [],
        'wd_instantid_training' : [],
        'wd_gpt_training' : [],
    }



    for category in tqdm(categories):

        sim_dreambooth_training = calculate_cos_similarity(image_embeddings['dreambooth_images'][category],
                                                           image_embeddings['training_images'][category])
        results['sim_dreambooth_training'].append(sim_dreambooth_training.cpu().item())
        

        sim_instantid_training = calculate_cos_similarity(image_embeddings['instant_id_images'][category],
                                                           image_embeddings['training_images'][category])
        results['sim_instantid_training'].append(sim_instantid_training.cpu().item())
        

        sim_gpt_training = calculate_cos_similarity(image_embeddings['gpt_4o_images'][category],
                                                    image_embeddings['training_images'][category])
        results['sim_gpt_training'].append(sim_gpt_training.cpu().item())
    
    for category in tqdm(categories):
        dream = image_embeddings['dreambooth_images'][category]
        instant = image_embeddings['instant_id_images'][category]
        gpt = image_embeddings['gpt_4o_images'][category]
        train = image_embeddings['training_images'][category]

        # Wasserstein
        wasserstein_results['wd_dreambooth_training'].append(calculate_wasserstein_distance(dream, train))
        wasserstein_results['wd_instantid_training'].append(calculate_wasserstein_distance(instant, train))
        wasserstein_results['wd_gpt_training'].append(calculate_wasserstein_distance(gpt, train))

    

    pd.DataFrame(results).to_csv(args.evaluation_report_path, index=False)

    wasserstein_report_path = args.evaluation_report_path.replace('.csv', '_wasserstein.csv')
    pd.DataFrame(wasserstein_results).to_csv(wasserstein_report_path, index=False)
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dreambooth_generated_images_folder',
        type=str,
        help='Path where lie the images generated by dreambooth',
        default='../output_images/dreambooth/'
    )
    parser.add_argument(
        '--instant_id_generated_images_folder',
        type=str,
        help='Path where lie the images generated by instant_id',
        default='../output_images/instant_id/'
    )
    parser.add_argument(
        '--gpt_4o_generated_paths',
        type=str,
        help='Path where lie the images generated by gpt 4o',
        default='../output_images/gpt_4o/'
    )
    parser.add_argument(
        '--training_dataset_path',
        type=str,
        help='Path where lie the images generated by dreambooth',
        default='../train_dataset_dreambooth/'
    )
    parser.add_argument(
        '--evaluation_report_path',
        type=str,
        help='Path where lie the images generated by dreambooth',
        default='report.csv'
    )
    args = parser.parse_args()
    main(args)

    



