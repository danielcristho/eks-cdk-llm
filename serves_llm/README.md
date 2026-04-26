# LLM on EKS: Serving with vLLM

Last year, I mentioned that I’m interested in learning MLOps and LLMs. At first it was just curiosity, but over time I wanted to actually try building something—not just reading about it.

This post is a small step in that direction: serving an LLM using [vLLM](https://github.com/vllm-project/vllm), deployed on [Amazon EKS](https://aws.amazon.com/eks), provisioned the infra using [AWS CDK](https://github.com/aws/aws-cdk), and wrapped into a simple chatbot using [Streamlit](https://github.com/streamlit/streamlit).

## TL;DR

- I’m exploring MLOps + LLMs by building a real project
- Using vLLM for inference
- Running on EKS with CDK
- Simple chatbot with Streamlit

## What We Tryna Build

The idea is simple: build a small chatbot powered by an LLM and run the model on Kubernetes.

I’m not focusing on training models here. I just want to understand how to serve an LLM properly.

The flow looks like this:

- User interacts with a chatbot (running locally)
- The chatbot sends a request to a vLLM API
- The model processes the request and returns a response
- The vLLM service runs on Amazon EKS

![That's not enough](https://res.cloudinary.com/diunivf9n/image/upload/v1777184871/not-enough-batman_jpvyc6.gif)

Aight Thanks for reading this post, hope you found something usefull 🚀