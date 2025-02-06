FROM node:18-slim

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

EXPOSE 3005

CMD ["npm", "run", "dev", "--", "-p", "3005"] 