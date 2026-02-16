import torch
import torch.nn as nn
from torchvision import models

class CNNEncoderAttention(nn.Module):
    """
    Convolutional encoder that extracts spatial feature maps suitable
    for attention-based image captioning models.

    This encoder uses a ResNet-50 backbone with optional ImageNet
    pretrained weights. Unlike standard global-feature encoders, the
    final pooling and classification layers are removed so that the
    spatial feature map is preserved. The resulting feature map is
    reshaped into a sequence of spatial feature vectors that can be
    consumed by an attention-based decoder.

    Args:
        pretrained (bool): If True, load ImageNet-pretrained weights for
            the ResNet-50 backbone.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the ResNet-50 backbone and freeze its parameters to
        prevent updates during training.
        """
        super().__init__()
        resnet = models.resnet50(
            models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        for p in resnet.parameters():
            p.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # keep spatial map
        self.feat_dim = 2048

    def forward(self, x):
        """
        Extract spatial feature sequences from a batch of images.

        The backbone produces a spatial feature map of shape
        (B, C, H, W). This map is reshaped into a sequence of spatial
        feature vectors of shape (B, H*W, C), where each position
        corresponds to a spatial region in the image.

        Args:
            x (Tensor): Input image batch of shape (batch_size, C, H, W).

        Returns:
            Tensor: Spatial feature tensor of shape
                (batch_size, num_regions, feat_dim), where
                num_regions = H * W.
        """
        feat = self.backbone(x)               # (B, 2048, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W).permute(0, 2, 1)  # (B, 49, 2048)
        return feat
    

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism for computing a context
    vector over encoder features conditioned on the decoder hidden state.

    The attention module projects both the encoder outputs and the
    decoder hidden state into a shared attention space, computes
    compatibility scores, and produces normalized attention weights
    (alpha). These weights are then used to compute a weighted sum of
    encoder features, resulting in a context vector that summarizes the
    most relevant spatial or temporal information for the current
    decoding step.

    Args:
        feat_dim (int): Dimensionality of encoder feature vectors.
        hidden_size (int): Size of the decoder hidden state.
        attn_dim (int): Dimensionality of the intermediate attention
            projection space.
    """

    def __init__(self, feat_dim, hidden_size, attn_dim=256):
        """
        Initialize the linear projection layers used to compute additive
        attention scores.
        """
        super().__init__()
        self.encoder_attn = nn.Linear(feat_dim, attn_dim)
        self.decoder_attn = nn.Linear(hidden_size, attn_dim)
        self.full_attn = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, hidden):
        """
        Compute the attention context vector and attention weights.

        Args:
            encoder_out (Tensor): Encoder output features of shape
                (batch_size, num_positions, feat_dim).
            hidden (Tensor): Current decoder hidden state of shape
                (batch_size, hidden_size).

        Returns:
            Tuple[Tensor, Tensor]:
                - context (Tensor): Attention-weighted context vector of
                  shape (batch_size, feat_dim).
                - alpha (Tensor): Attention weights over encoder
                  positions of shape (batch_size, num_positions).
        """
        # encoder_out: (B, num_pixels, feat_dim)
        # hidden: (B, hidden_size)

        enc = self.encoder_attn(encoder_out)          # (B, num_pixels, attn_dim)
        dec = self.decoder_attn(hidden).unsqueeze(1)  # (B, 1, attn_dim)

        scores = torch.tanh(enc + dec)
        scores = self.full_attn(scores).squeeze(-1)   # (B, num_pixels)

        alpha = torch.softmax(scores, dim=1)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)  # (B, feat_dim)

        return context, alpha
    
class AttentionDecoder(nn.Module):
    """
    Attention-based LSTM decoder for image caption generation.

    At each decoding step, the decoder computes a context vector using
    Bahdanau (additive) attention over the encoder's spatial feature
    outputs. The context vector is concatenated with the current word
    embedding and passed to an LSTMCell to update the hidden state. The
    updated hidden state is then projected to vocabulary logits for
    token prediction.

    Args:
        vocab_size (int): Size of the output vocabulary.
        feat_dim (int): Dimensionality of encoder feature vectors.
        embedding_size (int): Dimensionality of token embeddings.
        hidden_size (int): Hidden state size of the LSTM decoder.
    """

    def __init__(self, vocab_size, feat_dim=2048, embedding_size=128, hidden_size=64):
        """
        Initialize embedding, attention module, LSTMCell, output layer,
        and layers used to initialize the decoder hidden state from
        encoder features.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = BahdanauAttention(feat_dim, hidden_size)

        self.lstm = nn.LSTMCell(embedding_size + feat_dim, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.init_h = nn.Linear(feat_dim, hidden_size)
        self.init_c = nn.Linear(feat_dim, hidden_size)

    def init_hidden(self, encoder_out):
        """
        Initialize the decoder hidden and cell states from encoder
        outputs by averaging spatial features and projecting them into
        the LSTM hidden space.

        Args:
            encoder_out (Tensor): Encoder output features of shape
                (batch_size, num_positions, feat_dim).

        Returns:
            Tuple[Tensor, Tensor]: Initial hidden state (h) and cell
            state (c), each of shape (batch_size, hidden_size).
        """
        mean_feat = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_feat))
        c = torch.tanh(self.init_c(mean_feat))
        return h, c

    def forward(self, encoder_out, captions):
        """
        Decode a sequence of caption tokens using attention over encoder
        features.

        For each time step, the decoder computes attention weights over
        encoder outputs, forms a context vector, concatenates it with the
        token embedding, updates the LSTM state, and produces vocabulary
        logits.

        Args:
            encoder_out (Tensor): Encoder feature tensor of shape
                (batch_size, num_positions, feat_dim).
            captions (Tensor): Input caption token indices of shape
                (batch_size, seq_len).

        Returns:
            Tensor: Vocabulary logits for each time step of shape
                (batch_size, seq_len, vocab_size).
        """
        B, T = captions.shape
        embeddings = self.embedding(captions)

        h, c = self.init_hidden(encoder_out)

        outputs = []

        for t in range(T):
            context, _ = self.attention(encoder_out, h)
            inp = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(inp, (h, c))
            out = self.fc(h)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs
    
class HybridModelAttention(nn.Module):
    """
    Hybrid image-captioning model using a CNN encoder with spatial
    features and an attention-based LSTM decoder.

    The encoder extracts spatial feature maps from input images, which
    are then fed to the attention-based decoder to generate caption
    sequences. The decoder attends to relevant spatial regions of the
    image at each time step when predicting the next token.

    Args:
        vocab_size (int): Size of the output vocabulary for the decoder.
    """

    def __init__(self, vocab_size):
        """
        Initialize the attention-based encoder and decoder modules.
        """
        super().__init__()
        self.encoder = CNNEncoderAttention()
        self.decoder = AttentionDecoder(vocab_size)

    def forward(self, images, captions):
        """
        Perform a forward pass of the hybrid attention-based model.

        Args:
            images (Tensor): Input image batch of shape
                (batch_size, C, H, W).
            captions (Tensor): Input caption token indices of shape
                (batch_size, seq_len).

        Returns:
            Tensor: Decoder output logits of shape
                (batch_size, seq_len, vocab_size), representing
                predicted scores for each token at each time step.
        """
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions)
        return outputs