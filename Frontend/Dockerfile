# ================================================================
# Stage 1: Cài đặt dependencies
# ================================================================
FROM node:20-alpine AS deps

WORKDIR /app

# Sao chép file package để tận dụng Docker layer caching
COPY package.json package-lock.json* ./
RUN npm ci --frozen-lockfile

# ================================================================
# Stage 2: Build production bundle
# ================================================================
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build Vite production
RUN npm run build

# ================================================================
# Stage 3: Serve với Nginx
# ================================================================
FROM nginx:1.25-alpine AS runtime

# Xoá cấu hình mặc định của nginx
RUN rm /etc/nginx/conf.d/default.conf

# Sao chép file cấu hình nginx tùy chỉnh
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Sao chép static files đã build
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
