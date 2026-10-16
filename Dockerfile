FROM python:3.12-slim

# Hugging Face Spaces runs containers as user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# 1) CPU-only PyTorch (the default build is huge because it includes CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Other libraries (copied first, so Docker can cache this layer)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Project code and the trained model
COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user models ./models
COPY --chown=user tests ./tests

# Hugging Face expects the app to listen on port 7860
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]