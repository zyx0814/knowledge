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
