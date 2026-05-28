# API 接口地址配置指南

## 📁 配置文件说明

### 1. 环境变量文件

#### 开发环境 (.env.development)
```env
VITE_API_BASE_URL=/api
VITE_PUBLIC_BASE_URL=/public
```

#### 生产环境 (.env.production)
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_PUBLIC_BASE_URL=http://localhost:8000/public
```

### 2. 配置文件 (src/config.js)
```javascript
const isDev = import.meta.env.DEV

const config = {
  apiBaseUrl: isDev 
    ? '/api' 
    : (window.__ENV__?.API_BASE_URL || 'http://localhost:8000/api'),
  
  publicBaseUrl: isDev 
    ? '/public' 
    : (window.__ENV__?.PUBLIC_BASE_URL || 'http://localhost:8000/public')
}

export default config
```

### 3. API 客户端 (src/api.js)
```javascript
import axios from 'axios'
import config from './config.js'

const apiClient = axios.create({
  baseURL: config.apiBaseUrl,
  timeout: 300000,
  headers: {
    'X-API-Key': '123456'
  }
})

const publicClient = axios.create({
  baseURL: config.publicBaseUrl,
  timeout: 300000
})

export { apiClient, publicClient, config }
```

## 🚀 不同环境的配置方式

### 方式一：开发环境（使用 Vite 代理）

**1. 修改 vite.config.js**
```javascript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',  // 修改为你的后端地址
      changeOrigin: true
    },
    '/public': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

**2. 无需修改环境变量文件**，保持默认即可

### 方式二：开发环境（不使用代理，直连后端）

**1. 修改 .env.development**
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_PUBLIC_BASE_URL=http://localhost:8000/public
```

**2. 修改 config.js**（可选，或者直接使用环境变量）

### 方式三：生产环境（环境变量）

**方法 A：通过构建时环境变量**
```bash
# Linux/Mac
export VITE_API_BASE_URL=https://your-api.com/api
export VITE_PUBLIC_BASE_URL=https://your-api.com/public
npm run build

# Windows PowerShell
$env:VITE_API_BASE_URL="https://your-api.com/api"
$env:VITE_PUBLIC_BASE_URL="https://your-api.com/public"
npm run build
```

**方法 B：修改 .env.production 文件**
```env
VITE_API_BASE_URL=https://your-api.com/api
VITE_PUBLIC_BASE_URL=https://your-api.com/public
```
然后执行 `npm run build`

**方法 C：运行时动态配置（推荐）**

创建一个 `env-config.js` 文件：
```javascript
window.__ENV__ = {
  API_BASE_URL: 'https://your-api.com/api',
  PUBLIC_BASE_URL: 'https://your-api.com/public'
}
```

在 `index.html` 中引入：
```html
<script src="env-config.js"></script>
<script type="module" src="/src/main.js"></script>
```

这样可以在部署后通过修改 `env-config.js` 来更改 API 地址，无需重新构建。

## 🔧 常见场景配置

### 场景 1：后端在本地 8000 端口

**开发环境**：
- 使用默认配置即可

**生产环境**：
```env
VITE_API_BASE_URL=http://localhost:8000/api
VITE_PUBLIC_BASE_URL=http://localhost:8000/public
```

### 场景 2：后端在远程服务器

**开发环境**（直连）：
```env
VITE_API_BASE_URL=http://192.168.1.100:8000/api
VITE_PUBLIC_BASE_URL=http://192.168.1.100:8000/public
```

或者使用代理（修改 vite.config.js）：
```javascript
proxy: {
  '/api': {
    target: 'http://192.168.1.100:8000',
    changeOrigin: true
  }
}
```

### 场景 3：使用 HTTPS 和域名

```env
VITE_API_BASE_URL=https://api.example.com/api
VITE_PUBLIC_BASE_URL=https://api.example.com/public
```

## 📝 修改 API Key

如果需要修改 API Key，编辑 `src/api.js`：

```javascript
const apiClient = axios.create({
  baseURL: config.apiBaseUrl,
  timeout: 300000,
  headers: {
    'X-API-Key': 'your-new-api-key'  // 修改这里
  }
})
```

同时需要同步修改后端配置 `backend/config/config.py`：
```python
API_KEY: str = "your-new-api-key"
```

## 🔍 验证配置

启动前端后，可以在浏览器控制台查看：

```javascript
// 检查配置
import { config } from './src/api.js'
console.log('API Base URL:', config.apiBaseUrl)
console.log('Public Base URL:', config.publicBaseUrl)
```

## 💡 最佳实践

1. **开发环境**：使用 Vite 代理，方便调试
2. **生产环境**：使用运行时配置（env-config.js），灵活部署
3. **版本控制**：`.env.development` 和 `.env.production` 可以提交到 Git
4. **敏感信息**：不要将生产环境的真实地址提交到公开仓库
