const isDev = import.meta.env.DEV

const config = {
  apiBaseUrl: isDev 
    ? '/api' 
    : (window.__ENV__?.API_BASE_URL || 'http://172.31.110.242:8000/api'),
  
  publicBaseUrl: isDev 
    ? '/public' 
    : (window.__ENV__?.PUBLIC_BASE_URL || 'http://172.31.110.242:8000/public')
}

export default config
