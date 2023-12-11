#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Importing Packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import pytorch_lightning as pl
from collections import OrderedDict
from torchtext import vocab # This package can give problems sometimes, it may be necessary to downgrade to a specific version
from pytorch_lightning.loggers import CSVLogger
from random import choice
import seaborn as sns
import random
from random import choice
import matplotlib.pyplot as plt
from sklearn import metrics
import torchmetrics
import enum
import csv
import os
import pickle
from sklearn.model_selection import train_test_split
from Bio import AlignIO
import math
import pathlib
import warnings
from conv_vae_model import ConvVAE
from reward_model import CNN

# Set up Amino Acid Dictionary of Indices
AAs = 'ACDEFGHIKLMNPQRSTVWY-' # setup torchtext vocab to map AAs to indices, usage is aa2ind(list(AAsequence))
WT = "MAGLRHTFVVADATLPDCPLVYASEGFYAMTGYGPDEVLGHNARFLQGEGTDPKEVQKIRDAIKKGEACSVRLLNYRKDGTPFWNLLTVTPIKTPDGRVSKFVGVQVDVTSKTEGKALA"
aa2ind = vocab.vocab(OrderedDict([(a, 1) for a in AAs]))
aa2ind.set_default_index(20) # set unknown charcterers to gap
    
# Loading VAE
def load_vae_model(checkpoint_path):
    """
    Load a ConvVAE model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
    Returns:
        ConvVAE: Loaded model.
    """
    vae_model = ConvVAE.load_from_checkpoint(checkpoint_path)
    return vae_model

# Loading reward model
def load_reward_model(checkpoint_path):
    """
    Load a reward model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the saved checkpoint file.
    Returns:
        ConvVAE (nn.Module)
    """
    reward_model = CNN.load_from_checkpoint(checkpoint_path)
    return reward_model

# Detect hamming distance between proteins
def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings, ignoring gaps,
       and return the positions of mutations along with the original and new amino acids."""
    if len(s1) != len(s2):
        raise ValueError("Sequences must be of the same length")

    distance = 0
    mutation_info = []
    
    # Filter out positions where either sequence has a gap
    filtered_pairs = [(i, el1, el2) for i, (el1, el2) in enumerate(zip(s1, s2)) if el1 != '-' and el2 != '-']

    for i, el1, el2 in filtered_pairs:
        if el1 != el2:
            distance += 1
            mutation_detail = {
                'pos': i+1,
                'orig_aa': el1,
                'mut_aa': el2
            }
            mutation_info.append(mutation_detail)
            
    return distance, mutation_info

# Define function for passing in tensor proteins through VAE encoder and decoder
def Using_VAE(VAE, batch):
    """
    Computes probabilities for a given batch of sequences using VAE
    Args:
        batch (torch.Tensor): A LongTensor of shape (b, L) with b = batch size and L = sequence length.
    Returns:
        probabilities (torch.FloatTensor):
    """
    with torch.no_grad():  # We do not want training to occur during scoring
        VAE.eval()
        torch.manual_seed(3)
        # Pass the batch through the model to get the z_mean, z_log_var, encoded, and decoded tensors
        z_mean, z_log_var, encoded, decoded = VAE(batch)
        VAE.train()
        
    return encoded, decoded

# Creating initial state (Inference with Pre-trained VAE)
def decoding(VAE, batch):
    """
    Computes probabilities matrices (states) for given batch of sequences using VAEs
    Args:
        batch (torch.Tensor): (b, latent_dim)
            A LongTensor of shape (b, latent_dim) with b = batch size and latent_dim = dimensions of latent space in VAE
        pre_trained_VAE (pytorch model): pretrained MSA VAE that remains frozen during training
    Returns:
        states (torch.FloatTensor): (b, dict, L)
            batch of state matrices for each protein with column corresponding to amino acid index of protein normalized to 1 (probabilities)
    """
    with torch.no_grad():  # We do not want training to occur during scoring
        VAE.eval()
        logits = VAE.decoder(batch)  # Use z_mean instead of encoded to remove stochastic reparameterization
        VAE.train()
    return logits

# Adding noise to CreiLOV representation in vae latent space to create dataset for RLXF
def generate_and_evaluate_mutants(vae_model, WT, AAs, scale=0.9111111111111111, num_samples=1000):
    torch.manual_seed(3)  # For reproducibility
    
    # Load parameters
    CreiLOV_representation = torch.load('./data/for_RLXF/CreiLOV_representation.pt')
    std_difference_from_best_single_mutant = np.load('./data/for_RLXF/singlemutant_std_est.npy')
    latent_dim = CreiLOV_representation.shape[0]

    # Generate noise
    noise = torch.normal(mean=0,
                         std=std_difference_from_best_single_mutant * scale,
                         size=(num_samples, latent_dim))

    # Create new latent representations by adding the noise to the original WT representation
    new_representations = CreiLOV_representation + noise

    # Send sequence into VAE to find average hamming distance
    decoded_SMs = decoding(vae_model, new_representations)

    # Identify the max values for each amino acid positions
    _, max_indices = torch.max(decoded_SMs, dim=1)

    # Create a reverse mapping manually based on the AAs string
    ind2aa = {i: aa for i, aa in enumerate(AAs)}

    # Convert the tensor to a list of indices and decode the sequences
    decoded_sequences = [''.join([ind2aa[idx] for idx in batch]) for batch in max_indices.tolist()]

    # # Send sequence into VAE to find average hamming distance
    # decoded_SMs = decoding(vae_model, new_representations)

    # # Ensure probabilities sum to 1 across amino acids for each position
    # normalized_probs = torch.nn.functional.softmax(decoded_SMs, dim=1)

    # # Sample from the probability distribution for each amino acid index in each protein
    # sampled_indices = torch.zeros(normalized_probs.size(0), normalized_probs.size(2)).long()
    # for i in range(normalized_probs.size(0)):
    #     for j in range(normalized_probs.size(2)):
    #         sampled_indices[i, j] = torch.multinomial(normalized_probs[i, :, j], 1).squeeze()

    # # # Checking the dimensions and type of sampled_indices
    # # print("sampled_indices")
    # # print(sampled_indices)
    # # print(sampled_indices.type())
    # # print(sampled_indices.size())

    # # Create a reverse mapping manually based on the AAs string
    # ind2aa = {i: aa for i, aa in enumerate(AAs)}

    # # Convert the tensor to a list of indices and decode the sequences
    # decoded_sequences = [''.join([ind2aa[idx] for idx in protein]) for protein in sampled_indices.tolist()]
    # # print(decoded_sequences)

    # Calculate metrics
    unique_sequences = set(decoded_sequences)
    num_unique_sequences = len(unique_sequences)
    hd_list = []
    unique_mutations = set()
    unique_positions = set()

    for decoded_sequence in decoded_sequences:
        hd, mutation_info = hamming_distance(WT, decoded_sequence)
        hd_list.append(hd)

        for mut in mutation_info:
            unique_mut = f"{mut['pos']}_{mut['orig_aa']}_{mut['mut_aa']}"
            unique_mutations.add(unique_mut)
            unique_positions.add(mut['pos'])

    # Compile metrics
    avg_hd = sum(hd_list) / len(hd_list) if hd_list else 0
    max_hd = max(hd_list) if hd_list else 0
    mutation_diversity = len(unique_mutations)
    pos_diversity = len(unique_positions)

    return new_representations, {
        'average_hamming_distance': avg_hd,
        'maximum_hamming_distance': max_hd,
        'mutation_diversity': mutation_diversity,
        'position_diversity': pos_diversity,
        'number_of_unique_sequences': num_unique_sequences}, decoded_sequences, hd_list

# # Generate dataset with close to hamming distance of 5 from CreiLOV
# def find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd=5, initial_scale=None, depth=0, closest_avg_hd=None, closest_dataset=None, closest_dataset_metrics=None):
#     # Initialize variables to track the closest dataset
#     # closest_dataset = None
#     # closest_dataset_metrics = None
#     closest_scale = initial_scale

#     if depth < 5:
#         if closest_avg_hd is not None:
#             if (4.9 <= closest_avg_hd <= 5.1):
#                 scale_adjustment = 0.05
#                 # print(scale_adjustment)
#             if (4.8 <= closest_avg_hd <= 5.2):
#                 scale_adjustment = 0.1
#                 # print(scale_adjustment)
#             else:
#                 scale_adjustment = 0.2
#                 # print(scale_adjustment)
#         else:
#             scale_adjustment = 0.05
#             # print(scale_adjustment)
    
#     else:
#         scale_adjustment = 0.05
        
#     # If initial_scale is provided, adjust scale factors around it
#     if initial_scale is not None:
#         if closest_avg_hd < 5.0:
#             lower_bound = closest_scale
#             upper_bound = closest_scale * (1 + scale_adjustment)
#         else:
#             lower_bound = closest_scale * (1 - scale_adjustment)
#             upper_bound = closest_scale

#         scale_factors = np.linspace(lower_bound, upper_bound, num=10)
#         print('New scale factors')
#         # print('New scale_factors:', scale_factors)
    
#     else:
#         # Define default scale factors to test if initial_scale is None
#         closest_avg_hd = float('inf')
#         scale_factors = np.linspace(0.9, 1.1, num=10)
#         # print('OG scale_factors:')
    
#     # Run the generation and evaluation 10 times with different scale factors
#     for scale in scale_factors:
#         dataset, metrics, _, _ = generate_and_evaluate_mutants(vae_model, WT, AAs, scale=scale, num_samples=num_samples)
#         avg_hd = metrics['average_hamming_distance']
#         # if depth > 10:
#         #     print(avg_hd)
        
#         # Check if this metrics is closer to the target average Hamming distance
#         if abs(avg_hd - target_hd) < abs(closest_avg_hd - target_hd):
#             closest_avg_hd = avg_hd # Save best current average hamming distance metric
#             closest_dataset = dataset # Save dataset with closest to desired average hamming distance metric
#             closest_dataset_metrics = metrics # Save metrics for dataset
#             closest_scale = scale # Save scale of noise to create dataset

#     # # Log the scale and HD after evaluating all scale factors
#     # print(f"After evaluating scale factors, closest scale: {closest_scale} with average HD: {closest_avg_hd}")

#     # If the depth is less than 10 and the closest average HD is not within the desired range, recurse
#     if depth < 20 and not (4.975 <= closest_avg_hd <= 5.025):
#         print(f"Closest scale: {closest_scale} with aver HD: {closest_avg_hd}")
#         return find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd, closest_scale, depth+1, closest_avg_hd, closest_dataset, closest_dataset_metrics)
#     else:
#         # Print the closest scale for debugging or logging purposes
#         print(f"Closest scale factor: {closest_scale} with aver HD: {closest_avg_hd}")
#         return closest_dataset, closest_dataset_metrics, closest_scale # Return the dataset, metrics, and scale factor

# ProtRepDataset is a data handling class for getting protein representations from .pt file
class ProtRepDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein representations from pre-trained VAE"""
    def __init__(self, rl_updated_vae, WT, AAs, num_samples=1000, target_hd=5, initial_scale=None, current_epoch=0, version=None):
        # Store parameters as instance attributes
        self.vae_model = rl_updated_vae
        self.WT = WT
        self.AAs = AAs
        self.num_samples = num_samples
        self.target_hd = target_hd
        self.initial_scale = None # Scale does not change much throughout training, this could be edited in the future for recursively updating the scale
        self.current_epoch = current_epoch
        self.version = version

        # Update the save_dir to include logger version information if available
        if self.version is not None and self.version is not None:
            version_path = f'version_{self.version}'
        else:
            version_path = 'version_unknown'
        self.save_dir = os.path.join('./data/for_RLXF/training_generated_data/vae', version_path)
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the directory exists
        
        # Initialize the dataset with data for the current epoch
        self.data, self.metrics, self.scale = self.create_data_for_epoch()
        self.save_data()

    def create_data_for_epoch(self):
        closest_dataset, closest_dataset_metrics, closest_scale = find_closest_average_hd(
            self.vae_model, self.WT, self.AAs, self.num_samples, self.target_hd, self.initial_scale)
        return closest_dataset, closest_dataset_metrics, closest_scale

    def save_data(self):
        # Where to save data
        data_filename = f'data_epoch_{self.current_epoch}.pt'
        metrics_and_scale_filename = f'metrics_and_scale_epoch_{self.current_epoch}.txt'
        data_path = os.path.join(self.save_dir, data_filename)
        metrics_and_scale_path = os.path.join(self.save_dir, metrics_and_scale_filename)

        # Saving data, metrics, and scale
        torch.save(self.data, data_path)

        # Saving metrics and scale to the same text file
        with open(metrics_and_scale_path, 'w') as file:
            for key, value in self.metrics.items():
                file.write(f"{key}: {value}\n")
            file.write(f"scale: {self.scale}\n")
        
        print(f"Data: {data_path}, metrics and scale: {metrics_and_scale_path}.")

    def __len__(self):
        # Return the number of items in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return the item at the given index
        return self.data[idx]

# Dataloader for RLXF with VAE
class ProtDataModule(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting for protein representations"""

    def __init__(self, rl_updated_vae, WT, AAs, batch_size, num_samples=1000, target_hd=5, initial_scale=None, current_epoch=0, version=None):
        super().__init__()
        self.rl_updated_vae = rl_updated_vae
        self.WT = WT
        self.AAs = AAs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.target_hd = target_hd
        self.initial_scale = None # Scale does not change much throughout training, this could be edited in the future for recursively updating the scale
        self.current_epoch = current_epoch
        self.version = version


    # This can help with loading data
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # Updated to pass the necessary arguments to ProtRepDataset
            self.train_ds = ProtRepDataset(
                self.rl_updated_vae, 
                self.WT, 
                self.AAs, 
                self.num_samples, 
                self.target_hd, 
                self.initial_scale,
                self.current_epoch,
                self.version
            )
            
    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

# ProtRepDataset is a data handling class for getting protein representations from .pt file
class ProtRepDataset_0(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein representations from pre-trained VAE"""

    def __init__(self, protein_reps):
        self.protein_reps = protein_reps

    def __getitem__(self, idx):
        rep = self.protein_reps[idx]  # Extract protein representation at index idx
        return rep  # Return protein representation

    def __len__(self):
        return len(self.protein_reps)

# ProtDataModule randomizes protein representation batches per epoch of PPO
class ProtDataModule_0(pl.LightningDataModule):
    """A PyTorch Lightning Data Module to handle data splitting"""

    def __init__(self, protein_reps, batch_size):
        # Call the __init__ method of the parent class
        super().__init__()

        # Store the batch size
        self.protein_reps = protein_reps
        self.batch_size = batch_size

    # This can help with loading data to GPU
    def prepare_data(self):
        pass

    # Assign datasets for use in dataloaders
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_ds = ProtRepDataset_0(self.protein_reps)
            
    def train_dataloader(self):
        return data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=8)

# Enforce MA- and -KALA to protein where MSA had high probability of gaps
def adjust_designs(mutant_designs):
    adjusted_sequences = []
    
    for seq in mutant_designs:
        # Convert string to list of characters for easier manipulation
        seq_list = list(seq)
        
        # Ensure the N terminus begins with Methionine (M) and Alanine (A)
        # Assuming 'M' is represented by '10' and 'A' is represented by '0'
        seq_list[0] = 'M'  # Replace with correct letter or code for Methionine
        seq_list[1] = 'A'  # Replace with correct letter or code for Alanine

        # Ensure the C terminus ends with KALA
        # Assuming 'K' is '8', 'A' is '0', 'L' is '9', 'A' is '0'
        seq_list[-4] = 'K'  # Replace with correct letter or code for Lysine
        seq_list[-3] = 'A'  # Replace with correct letter or code for Alanine
        seq_list[-2] = 'L'  # Replace with correct letter or code for Leucine
        seq_list[-1] = 'A'  # Replace with correct letter or code for Alanine

        # Revert the list of characters back to a string
        adjusted_seq = ''.join(seq_list)
        adjusted_sequences.append(adjusted_seq)
    return adjusted_sequences

# SeqDataset is a data handling class. I convert amino acid sequences to torch tensors for model input
class SeqDataset(torch.utils.data.Dataset):
    """A custom PyTorch dataset for protein sequence-function data"""

    def __init__(self, data_frame):
        self.data_df = data_frame

    def __getitem__(self, idx):
        sequence = torch.tensor(aa2ind(list(self.data_df.Sequence.iloc[idx]))) # Extract sequence at index idx
        return sequence

    def __len__(self):
        return len(self.data_df)

# Score designs from rl_updated_vae
def convert_and_score_sequences(sequences_list, reward_model):
    # Convert the list of sequences into a DataFrame
    sequences_df = pd.DataFrame(sequences_list, columns=['Sequence'])

    # Initialize your custom dataset with the DataFrame
    sequence_dataset = SeqDataset(sequences_df)

    # Initialize the DataLoader
    sequence_loader = data_utils.DataLoader(sequence_dataset, batch_size=1, shuffle=False)

    # Prepare a tensor to store the scores
    scores_tensor = torch.zeros(len(sequences_list), dtype=torch.float32)

    # Ensure the model is in evaluation mode
    reward_model.eval()

    # No need for gradients
    with torch.no_grad():
        for i, sequence_tensor in enumerate(sequence_loader):
            # The sequence_tensor is a batch with a single sequence
            # If the reward_model expects a single sequence, you might need to squeeze it
            sequence_tensor = sequence_tensor.squeeze(0)

            # Predict the score for the sequence
            score = reward_model.predict(sequence_tensor.unsqueeze(0))[0][0]  # Add batch dimension back if necessary
            scores_tensor[i] = score

    return scores_tensor.tolist()  # Convert to list if needed

def save_metrics_to_csv(version, metrics):
    # Define the directory path and file path for the metrics
    dir_path = f'./designs/vae_designs/version_{version}'
    os.makedirs(dir_path, exist_ok=True)  # Create the directory if it does not exist
    metrics_file_path = os.path.join(dir_path, 'mutant_metrics.csv')
    
    # Save metrics to CSV file
    with open(metrics_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header with the keys of the metrics dictionary
        writer.writerow(metrics.keys())
        # Write the values of the metrics dictionary
        writer.writerow(metrics.values())

    print(f"Metrics for mutants saved to {metrics_file_path}")

def identify_mutations(wt_sequence, mutant_sequence):
    mutations = []
    for i, (wt_res, mut_res) in enumerate(zip(wt_sequence, mutant_sequence)):
        if wt_res != mut_res:
            mutations.append(f"{wt_res}{i+1}{mut_res}")  # Assuming 1-based numbering
    return ', '.join(mutations)

def save_sorted_designs_to_csv(version, wt_sequence, adjusted_mutant_designs, scores):
    # Sort designs by scores
    sorted_designs_scores = sorted(zip(adjusted_mutant_designs, scores), key=lambda x: x[1], reverse=True)
    sorted_designs, sorted_scores = zip(*sorted_designs_scores)

    # Define the directory path based on the version
    dir_path = f'./designs/vae_designs/version_{version}'
    os.makedirs(dir_path, exist_ok=True)

    # Save sorted designs, mutations, and scores to CSV file
    designs_file_path = os.path.join(dir_path, 'sorted_mutant_designs_scores_mutations.csv')
    with open(designs_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Adjusted Design', 'Score', 'Mutations'])  # Write the header
        for design, score in zip(sorted_designs, sorted_scores):
            mutations = identify_mutations(wt_sequence, design)
            writer.writerow([design, score, mutations])

    print(f"Mutant designs sorted by/with scores and mutations data saved to {designs_file_path}")

# Generate dataset with close to hamming distance of 5 from CreiLOV
def find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd=5, initial_scale=None, depth=0, closest_avg_hd=None, closest_dataset=None, closest_dataset_metrics=None):
    # Initialize variables to track the closest dataset
    # closest_dataset = None
    # closest_dataset_metrics = None
    closest_scale = initial_scale

    if depth < 5:
        if closest_avg_hd is not None:
            if (4.96 <= closest_avg_hd <= 5.04):
                scale_adjustment = 0.002
                # print(scale_adjustment)
            if (4.9 <= closest_avg_hd <= 5.1):
                scale_adjustment = 0.01
                # print(scale_adjustment)
            else:
                scale_adjustment = 0.05
                # print(scale_adjustment)
        else:
            scale_adjustment = 0.005
            # print(scale_adjustment)
    
    else:
        scale_adjustment = 0.025
        
    # If initial_scale is provided, adjust scale factors around it
    if initial_scale is not None:
        if closest_avg_hd < 5.0:
            lower_bound = closest_scale
            upper_bound = closest_scale * (1 + scale_adjustment)
        else:
            lower_bound = closest_scale * (1 - scale_adjustment)
            upper_bound = closest_scale

        scale_factors = np.linspace(lower_bound, upper_bound, num=20*(depth+1))
        # print('New scale factors')
        # print('New scale_factors:', scale_factors)
    
    else:
        # Define default scale factors to test if initial_scale is None
        closest_avg_hd = float('inf')
        scale_factors = np.linspace(0.9, 1, num=20*(depth+1))
        # print('OG scale_factors:')
    
    # Run the generation and evaluation 10 times with different scale factors
    for scale in scale_factors:
        dataset, metrics, _, _ = generate_and_evaluate_mutants(vae_model, WT, AAs, scale=scale, num_samples=num_samples)
        avg_hd = metrics['average_hamming_distance']
        # if depth > 10:
        #     print(avg_hd)
        
        # Check if this metrics is closer to the target average Hamming distance
        if abs(avg_hd - target_hd) < abs(closest_avg_hd - target_hd):
            closest_avg_hd = avg_hd # Save best current average hamming distance metric
            closest_dataset = dataset # Save dataset with closest to desired average hamming distance metric
            closest_dataset_metrics = metrics # Save metrics for dataset
            closest_scale = scale # Save scale of noise to create dataset

    # # Log the scale and HD after evaluating all scale factors
    # print(f"After evaluating scale factors, closest scale: {closest_scale} with average HD: {closest_avg_hd}")

    # If the depth is less than 10 and the closest average HD is not within the desired range, recurse
    if depth < 10 and not (4.975 <= closest_avg_hd <= 5.025):
        print(f"Closest scale: {closest_scale} with aver HD: {closest_avg_hd}")
        return find_closest_average_hd(vae_model, WT, AAs, num_samples, target_hd, closest_scale, depth+1, closest_avg_hd, closest_dataset, closest_dataset_metrics)
    else:
        # Print the closest scale for debugging or logging purposes
        print(f"Closest scale factor: {closest_scale} with aver HD: {closest_avg_hd}")
        return closest_dataset, closest_dataset_metrics, closest_scale # Return the dataset, metrics, and scale factor

def count_mutations(mutation_str):
    """Count the number of mutations in the mutation string."""
    return len(mutation_str.split(','))

# Load Data from MSA for ConvVAE format
def get_msa_from_fasta(filename):
    import Bio.SeqIO
    with open(filename, "rt") as fh: 
        return [r[1] for r in Bio.SeqIO.FastaIO.SimpleFastaParser(fh)]

# One-hot encode a single sequence
def one_hot_encode(sequence, aa2ind, max_length):
    # Pad or truncate the sequence to the desired length
    sequence = sequence[:max_length] + '-' * (max_length - len(sequence))
    # Create a one-hot encoded matrix where each row corresponds to an amino acid
    one_hot_matrix = np.zeros((max_length, len(aa2ind)), dtype=np.int32)
    for position, amino_acid in enumerate(sequence):
        index = aa2ind.get(amino_acid, aa2ind['-'])  # Get index or use index of gap ('-') if amino acid not found
        one_hot_matrix[position, index] = 1
    return one_hot_matrix
    
# One-hot encode all sequences in the MSA
def one_hot_encode_msa(msa, aa2ind, max_length):
    # Initialize the array to hold the one-hot encoded sequences
    one_hot_msa = np.zeros((len(msa), max_length, len(aa2ind)), dtype=np.int32)
    for idx, sequence in enumerate(msa):
        one_hot_msa[idx] = one_hot_encode(sequence, aa2ind, max_length)
    return one_hot_msa


