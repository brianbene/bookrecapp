import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import re

class BookDataset(Dataset):
    """
    Dataset class for loading books for training.
    """

    def __init__(self, summaries, metadata, titles, tokenizer):
        self.summaries = summaries
        self.metadata = torch.FloatTensor(metadata)
        self.titles = titles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.summaries)

    def __getitem__(self, idx):
        # Convert book summary to tokens that BERT can understand
        summary = str(self.summaries[idx])
        encoding = self.tokenizer(
            summary,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'metadata': self.metadata[idx],
            'title': self.titles[idx]
        }


class BookVAE(nn.Module):
    def __init__(self, metadata_dim, latent_dim=32):
        super(BookVAE, self).__init__()

        self.latent_dim = latent_dim
        self.bert_dim = 768  # BERT embedding size
        self.metadata_dim = metadata_dim
        self.input_dim = self.bert_dim + metadata_dim

        # Load BERT model
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False

        # Encoder: compress book info to latent space
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(128, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(128, latent_dim)  # Variance

        # Decoder: reconstruct book info from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.input_dim),
        )

    def get_bert_embeddings(self, input_ids, attention_mask):
        """Get BERT embeddings for book summaries."""
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token embedding (first token)
            return outputs.last_hidden_state[:, 0, :]

    def encode(self, bert_embeddings, metadata):
        """Encode book info to latent space."""
        x = torch.cat([bert_embeddings, metadata], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent space back to book info."""
        return self.decoder(z)

    def forward(self, input_ids, attention_mask, metadata):
        """Full forward pass through the VAE."""
        # Get BERT embeddings
        bert_embeddings = self.get_bert_embeddings(input_ids, attention_mask)

        # Encode to latent space
        mu, logvar = self.encode(bert_embeddings, metadata)
        z = self.reparameterize(mu, logvar)

        # Decode back
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar, bert_embeddings


def vae_loss(reconstructed, original, mu, logvar):

    # How well did we reconstruct the original?
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='mean')

    # Keep latent space well-behaved (KL divergence)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    total_loss = recon_loss + 0.1 * kl_loss  # Weight KL loss less

    return total_loss, recon_loss, kl_loss


def clean_text(text):
    """Clean book summaries."""
    if pd.isna(text):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def prepare_metadata(df):

    print("Preparing metadata features...")

    metadata_features = []

    # Numerical features
    for col in ['page_count', 'average_rating', 'ratings_count']:
        values = pd.to_numeric(df[col], errors='coerce').fillna(0)
        metadata_features.append(values.values.reshape(-1, 1))

    # Publication year
    pub_years = pd.to_datetime(df['published_date'], errors='coerce').dt.year
    pub_years = pub_years.fillna(2000).astype(int)
    metadata_features.append(pub_years.values.reshape(-1, 1))

    # Genre encoding
    df['primary_genre'] = df['genres'].fillna('unknown').apply(
        lambda x: x.split(',')[0].strip().lower() if x != 'unknown' else 'unknown'
    )

    genre_encoder = LabelEncoder()
    genre_encoded = genre_encoder.fit_transform(df['primary_genre'])
    n_genres = len(genre_encoder.classes_)
    genre_onehot = np.eye(n_genres)[genre_encoded]
    metadata_features.append(genre_onehot)

    # Combine all metadata
    metadata_combined = np.hstack(metadata_features)

    # Scale the features
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata_combined)

    print(f"Metadata features shape: {metadata_scaled.shape}")

    return metadata_scaled, scaler, genre_encoder


def train_vae(df, metadata_features, epochs=20, batch_size=8):

    print("Training VAE model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split data
    train_idx, val_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = BookDataset(
        df.iloc[train_idx]['summary_clean'].values,
        metadata_features[train_idx],
        df.iloc[train_idx]['title'].values,
        tokenizer
    )

    val_dataset = BookDataset(
        df.iloc[val_idx]['summary_clean'].values,
        metadata_features[val_idx],
        df.iloc[val_idx]['title'].values,
        tokenizer
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = BookVAE(metadata_dim=metadata_features.shape[1], latent_dim=32)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)

            optimizer.zero_grad()

            reconstructed, mu, logvar, original = model(input_ids, attention_mask, metadata)
            original_combined = torch.cat([original, metadata], dim=1)

            loss, recon_loss, kl_loss = vae_loss(reconstructed, original_combined, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                metadata = batch['metadata'].to(device)

                reconstructed, mu, logvar, original = model(input_ids, attention_mask, metadata)
                original_combined = torch.cat([original, metadata], dim=1)

                loss, _, _ = vae_loss(reconstructed, original_combined, mu, logvar)
                val_loss += loss.item()

        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model, tokenizer


def create_book_embeddings(model, df, metadata_features, tokenizer):

    print("Creating book embeddings...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    embeddings = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_summaries = df.iloc[i:batch_end]['summary_clean'].values
            batch_metadata = metadata_features[i:batch_end]

            # Tokenize
            batch_encodings = tokenizer(
                batch_summaries.tolist(),
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )

            input_ids = batch_encodings['input_ids'].to(device)
            attention_mask = batch_encodings['attention_mask'].to(device)
            metadata_tensor = torch.FloatTensor(batch_metadata).to(device)

            # Get BERT embeddings and encode to latent space
            bert_embeddings = model.get_bert_embeddings(input_ids, attention_mask)
            mu, _ = model.encode(bert_embeddings, metadata_tensor)

            embeddings.append(mu.cpu().numpy())

    return np.vstack(embeddings)


def main():
    """
    Main function that trains the VAE model.
    """
    print("Starting VAE training...")

    # Load dataset
    try:
        dataset_file = os.path.join(DATA_PATH, "book_dataset.csv")
        df = pd.read_csv(dataset_file)
        print(f"Loaded {len(df)} books")
    except FileNotFoundError:
        print("❌ Error: book_dataset.csv not found!")
        print("Please run collect_book_data.py first")
        return

    # Clean summaries
    df['summary_clean'] = df['summary'].apply(clean_text)
    df = df[df['summary_clean'].str.len() >= 100]

    print(f"After cleaning: {len(df)} books")

    # Prepare metadata
    metadata_features, scaler, genre_encoder = prepare_metadata(df)

    # Train VAE
    model, tokenizer = train_vae(df, metadata_features)

    # Create book embeddings
    book_embeddings = create_book_embeddings(model, df, metadata_features, tokenizer)

    # Save everything
    print("Saving model and components...")

    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'bert_dim': model.bert_dim,
        'metadata_dim': model.metadata_dim,
        'latent_dim': model.latent_dim
    }, os.path.join(MODELS_PATH, 'book_vae_model.pth'))

    # Save preprocessing components
    components = {
        'tokenizer': tokenizer,
        'metadata_scaler': scaler,
        'genre_encoder': genre_encoder,
        'df': df
    }

    with open(os.path.join(MODELS_PATH, 'vae_preprocessing.pkl'), 'wb') as f:
        pickle.dump(components, f)

    # Save embeddings
    np.save(os.path.join(MODELS_PATH, 'book_embeddings.npy'), book_embeddings)

    print("✅ VAE training complete!")
    print(f"✅ Book embeddings saved: {book_embeddings.shape}")
    print("✅ All files saved successfully")



if __name__ == "__main__":
    main()