

import streamlit as st

import torch
import torch.nn as nn

import numpy as np

from transformers import BertTokenizer


# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title(
    "AI Text-to-Image Generation System"
)

st.write(
    "GAN + Attention + HuggingFace + Streamlit"
)

# -----------------------------------
# TOKENIZER
# -----------------------------------

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# -----------------------------------
# SELF ATTENTION
# -----------------------------------

class SelfAttention(nn.Module):

    def __init__(self, in_dim):

        super(SelfAttention, self).__init__()

        self.query = nn.Linear(
            in_dim,
            in_dim
        )

        self.key = nn.Linear(
            in_dim,
            in_dim
        )

        self.value = nn.Linear(
            in_dim,
            in_dim
        )

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        attention = torch.softmax(

            torch.matmul(
                Q,
                K.T
            ),

            dim=-1
        )

        out = torch.matmul(
            attention,
            V
        )

        return out


# -----------------------------------
# SHAPE GENERATOR
# -----------------------------------

class ShapeGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            2,
            10
        )

        self.attention = SelfAttention(10)

        self.fc = nn.Sequential(

            nn.Linear(20,128),

            nn.ReLU(),

            nn.Linear(128,784),

            nn.Tanh()
        )

    def forward(self, noise, labels):

        embed = self.embedding(labels)

        attended = self.attention(embed)

        x = torch.cat(
            [noise, attended],
            dim=1
        )

        img = self.fc(x)

        img = img.view(-1,28,28)

        return img


# -----------------------------------
# MEDICAL GENERATOR
# -----------------------------------

class MedicalGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            2,
            10
        )

        self.attention = SelfAttention(10)

        self.fc = nn.Sequential(

            nn.Linear(20,256),

            nn.ReLU(),

            nn.Linear(256,784),

            nn.Tanh()
        )

    def forward(self, noise, labels):

        embed = self.embedding(labels)

        attended = self.attention(embed)

        x = torch.cat(
            [noise, attended],
            dim=1
        )

        img = self.fc(x)

        img = img.view(-1,28,28)

        return img


shape_generator = ShapeGenerator()

medical_generator = MedicalGenerator()


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title(
    "Navigation"
)

option = st.sidebar.selectbox(

    "Choose Task",

    [

        "Shape Generator",

        "Medical Generator"

    ]
)

# -----------------------------------
# SHAPE SECTION
# -----------------------------------

if option == "Shape Generator":

    st.header(
        "GAN Shape Generator"
    )

    shape = st.selectbox(

        "Choose Shape",

        [

            "Circle",

            "Square"

        ]
    )

    if st.button("Generate Shape"):

        if shape == "Circle":

            label = torch.tensor([0])

        else:

            label = torch.tensor([1])

        noise = torch.randn(1,10)

        generated = shape_generator(

            noise,
            label

        ).detach().numpy()[0]

        st.image(

            generated,

            caption=f"Generated {shape}",

            clamp=True
        )

        st.success(
            "Shape Generated Successfully!"
        )


# -----------------------------------
# MEDICAL SECTION
# -----------------------------------

if option == "Medical Generator":

    st.header(
        "Medical Text-to-Image Generation"
    )

    text = st.text_input(

        "Enter Medical Text",

        "pneumonia chest xray"

    )

    if st.button("Generate Medical Image"):

        tokens = tokenizer.tokenize(text)

        st.write(
            "Tokens:",
            tokens
        )

        if "pneumonia" in text.lower():

            label = torch.tensor([1])

        else:

            label = torch.tensor([0])

        noise = torch.randn(1,10)

        generated = medical_generator(

            noise,
            label

        ).detach().numpy()[0]

        st.image(

            generated,

            caption="Generated Medical Image",

            clamp=True
        )

        st.success(
            "Medical Image Generated!"
        )

# -----------------------------------
# FOOTER
# -----------------------------------

st.write("---")

st.write(
    "Built using PyTorch, GANs, Self-Attention, HuggingFace Transformers, and Streamlit."
)

