# Stage 1: Build the Obfuscated Frontend
FROM node:20-alpine AS builder
WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source configuration and run build
COPY src/ ./src/
COPY public/ ./public/
# FIX: Ensure assets folder is imported into the Docker build context
COPY assets/ ./assets/
COPY build.js ./
RUN npm run build

# Stage 2: Serve with Nginx (Minimal Production Image)
FROM nginx:alpine
# Only the compiled /dist artifacts are copied. Source code is left behind.
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]