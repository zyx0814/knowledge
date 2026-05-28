<template>
  <div class="app-container">
    <el-header height="60px" class="header">
      <h1>本地知识库测试平台</h1>
      <el-menu :default-active="activeTab" mode="horizontal" @select="handleTabChange">
        <el-menu-item index="home">首页</el-menu-item>
        <el-menu-item index="batch-import">批量导入</el-menu-item>
        <el-menu-item index="semantic-search">语义搜索</el-menu-item>
        <el-menu-item index="image-search">以图搜图</el-menu-item>
        <el-menu-item index="face-search">人脸搜索</el-menu-item>
        <el-menu-item index="files">文件管理</el-menu-item>
      </el-menu>
    </el-header>

    <el-main class="main">
      <div v-if="activeTab === 'home'" class="home">
        <el-card class="welcome-card">
          <template #header>
            <div class="card-header">
              <span>欢迎使用本地知识库测试平台</span>
            </div>
          </template>
          <div class="welcome-content">
            <p>本平台用于测试本地知识库系统的各项功能，包括批量导入、语义搜索、以图搜图、人脸搜索等。</p>
            <el-row :gutter="20">
              <el-col :span="6">
                <el-card shadow="hover" class="feature-card" @click="activeTab = 'batch-import'">
                  <div class="feature-icon">📁</div>
                  <h4>批量导入</h4>
                  <p>选定本地目录批量导入文件</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="feature-card" @click="activeTab = 'semantic-search'">
                  <div class="feature-icon">🔍</div>
                  <h4>语义搜索</h4>
                  <p>基于向量相似度的智能搜索</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="feature-card" @click="activeTab = 'image-search'">
                  <div class="feature-icon">🖼️</div>
                  <h4>以图搜图</h4>
                  <p>上传图片搜索相似图片</p>
                </el-card>
              </el-col>
              <el-col :span="6">
                <el-card shadow="hover" class="feature-card" @click="activeTab = 'face-search'">
                  <div class="feature-icon">👤</div>
                  <h4>人脸搜索</h4>
                  <p>基于人脸特征的图片搜索</p>
                </el-card>
              </el-col>
            </el-row>
          </div>
        </el-card>
      </div>

      <div v-if="activeTab === 'batch-import'" class="batch-import">
        <el-card class="import-card">
          <template #header>
            <div class="card-header">
              <span>批量导入文件</span>
            </div>
          </template>
          <el-form :model="importForm" label-width="100px">
            <el-form-item label="目录路径">
              <el-input v-model="importForm.directory" placeholder="请输入本地目录的绝对路径" style="width: 500px;"></el-input>
            </el-form-item>
            <el-form-item label="区域ID">
              <el-input v-model="importForm.rid" placeholder="可选，用于分组"></el-input>
            </el-form-item>
            <el-form-item label="分组ID">
              <el-input-number v-model="importForm.gid" :min="0" placeholder="可选"></el-input-number>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="startBatchImportAsync" :loading="importing" style="margin-right: 10px;">
                异步导入（推荐，不阻塞）
              </el-button>
              <el-button @click="startBatchImport" :loading="importing">同步导入</el-button>
            </el-form-item>
          </el-form>
          
          <div v-if="importProgress.show" class="import-progress">
            <el-progress :percentage="importProgress.percentage" :status="importProgress.status"></el-progress>
            <p>{{ importProgress.message }}</p>
            <p v-if="currentTaskId" style="font-size: 12px; color: #999;">
              任务ID: {{ currentTaskId }}
            </p>
          </div>
        </el-card>
        
        <el-card class="task-card" style="margin-top: 20px;">
          <template #header>
            <div class="card-header">
              <span>任务列表</span>
              <el-button size="small" @click="refreshTasks">刷新</el-button>
            </div>
          </template>
          <el-table :data="tasks" style="width: 100%">
            <el-table-column prop="id" label="任务ID" width="280"></el-table-column>
            <el-table-column prop="type" label="类型" width="120"></el-table-column>
            <el-table-column label="状态" width="100">
              <template #default="scope">
                <el-tag :type="getTaskStatusType(scope.row.status)">
                  {{ getTaskStatusText(scope.row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="进度" width="150">
              <template #default="scope">
                <el-progress :percentage="scope.row.progress" :stroke-width="10"></el-progress>
              </template>
            </el-table-column>
            <el-table-column prop="message" label="消息"></el-table-column>
            <el-table-column label="操作" width="120">
              <template #default="scope">
                <el-button size="small" @click="viewTaskDetail(scope.row)">详情</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <div v-if="activeTab === 'semantic-search'" class="semantic-search">
        <el-card class="search-card">
          <template #header>
            <div class="card-header">
              <span>语义搜索</span>
            </div>
          </template>
          <el-form :inline="true" :model="semanticSearchForm" class="search-form">
            <el-form-item label="搜索词">
              <el-input v-model="semanticSearchForm.query" placeholder="请输入搜索内容" style="width: 400px;"></el-input>
            </el-form-item>
            <el-form-item label="文件类型">
              <el-select v-model="semanticSearchForm.fileType" placeholder="全部">
                <el-option label="全部" value=""></el-option>
                <el-option label="文档" value="document"></el-option>
                <el-option label="图片" value="image"></el-option>
                <el-option label="视频" value="video"></el-option>
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="doSemanticSearch" :loading="searching">搜索</el-button>
            </el-form-item>
          </el-form>

          <div v-if="semanticResults.length > 0" class="search-results">
            <h3>搜索结果 ({{ semanticResults.length }} 条)</h3>
            <el-row :gutter="20">
              <el-col :span="8" v-for="(result, index) in semanticResults" :key="index">
                <el-card shadow="hover" class="result-card">
                  <div class="result-type">
                    <el-tag :type="getTypeTagType(result.type)">{{ getTypeLabel(result.type) }}</el-tag>
                    <el-tag size="small" type="info">相似度: {{ (result.score * 100).toFixed(2) }}%</el-tag>
                  </div>
                  <h4>{{ result.name || result.file_info?.name || '未知文件' }}</h4>
                  <p class="result-content">{{ result.content || '无内容预览' }}</p>
                  <div v-if="result.type === 'image' || result.file_info?.type === 'image'" class="result-image">
                    <el-image 
                      :src="getImagePreviewUrl(result)" 
                      fit="cover" 
                      style="width: 100%; height: 150px;"
                      :preview-src-list="[getImagePreviewUrl(result)]"
                      lazy>
                      <template #error>
                        <div class="image-slot">
                          <el-icon><Picture /></el-icon>
                        </div>
                      </template>
                    </el-image>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          <el-empty v-else-if="semanticSearched" description="未找到相关内容"></el-empty>
        </el-card>
      </div>

      <div v-if="activeTab === 'image-search'" class="image-search">
        <el-card class="search-card">
          <template #header>
            <div class="card-header">
              <span>以图搜图</span>
            </div>
          </template>
          <el-form :model="imageSearchForm" class="search-form">
            <el-form-item label="上传图片">
              <el-upload
                class="image-uploader"
                :show-file-list="false"
                :on-change="handleImageChange"
                :auto-upload="false"
                accept="image/*">
                <img v-if="imageSearchForm.imageUrl" :src="imageSearchForm.imageUrl" class="uploaded-image" />
                <el-icon v-else class="image-uploader-icon"><Plus /></el-icon>
              </el-upload>
            </el-form-item>
            <el-form-item label="分组ID">
              <el-input-number v-model="imageSearchForm.gid" :min="0" placeholder="可选"></el-input-number>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="doImageSearch" :loading="imageSearching" :disabled="!imageSearchForm.imageUrl">搜索</el-button>
            </el-form-item>
          </el-form>

          <div v-if="imageSearchResults.length > 0" class="search-results">
            <h3>搜索结果 ({{ imageSearchResults.length }} 条)</h3>
            <div class="comparison-section">
              <h4>查询图片</h4>
              <el-image :src="imageSearchForm.imageUrl" style="width: 200px; height: 200px;" fit="contain"></el-image>
            </div>
            <el-row :gutter="20">
              <el-col :span="8" v-for="(result, index) in imageSearchResults" :key="index">
                <el-card shadow="hover" class="result-card">
                  <div class="result-type">
                    <el-tag :type="getTypeTagType(result.type || result.file_info?.type)">
                      {{ result.search_type === 'face' ? '人脸匹配' : '图片匹配' }}
                    </el-tag>
                    <el-tag size="small" type="info">相似度: {{ ((result.score || result.similarity) * 100).toFixed(2) }}%</el-tag>
                  </div>
                  <h4>{{ result.name || result.file_info?.name || '未知文件' }}</h4>
                  <div class="result-image">
                    <el-image 
                      :src="getImagePreviewUrl(result)" 
                      fit="cover" 
                      style="width: 100%; height: 200px;"
                      :preview-src-list="[getImagePreviewUrl(result)]"
                      lazy>
                      <template #error>
                        <div class="image-slot">
                          <el-icon><Picture /></el-icon>
                        </div>
                      </template>
                    </el-image>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          <el-empty v-else-if="imageSearched" description="未找到相似图片"></el-empty>
        </el-card>
      </div>

      <div v-if="activeTab === 'face-search'" class="face-search">
        <el-card class="search-card">
          <template #header>
            <div class="card-header">
              <span>人脸搜索</span>
            </div>
          </template>
          <el-form :model="faceSearchForm" class="search-form">
            <el-form-item label="上传图片">
              <el-upload
                class="image-uploader"
                :show-file-list="false"
                :on-change="handleFaceImageChange"
                :auto-upload="false"
                accept="image/*">
                <img v-if="faceSearchForm.imageUrl" :src="faceSearchForm.imageUrl" class="uploaded-image" />
                <el-icon v-else class="image-uploader-icon"><Plus /></el-icon>
              </el-upload>
            </el-form-item>
            <el-form-item label="分组ID">
              <el-input-number v-model="faceSearchForm.gid" :min="0" placeholder="可选"></el-input-number>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="doFaceSearch" :loading="faceSearching" :disabled="!faceSearchForm.imageUrl">搜索</el-button>
            </el-form-item>
          </el-form>

          <div v-if="faceSearchResults.length > 0" class="search-results">
            <h3>搜索结果 ({{ faceSearchResults.length }} 条)</h3>
            <div class="comparison-section">
              <h4>查询图片</h4>
              <el-image :src="faceSearchForm.imageUrl" style="width: 200px; height: 200px;" fit="contain"></el-image>
            </div>
            <el-row :gutter="20">
              <el-col :span="8" v-for="(result, index) in faceSearchResults" :key="index">
                <el-card shadow="hover" class="result-card">
                  <div class="result-type">
                    <el-tag type="danger">人脸匹配</el-tag>
                    <el-tag size="small" type="info">相似度: {{ ((result.score || result.similarity) * 100).toFixed(2) }}%</el-tag>
                  </div>
                  <h4>{{ result.file_info?.name || '未知文件' }}</h4>
                  <div class="result-image">
                    <el-image 
                      :src="getImagePreviewUrl(result)" 
                      fit="cover" 
                      style="width: 100%; height: 200px;"
                      :preview-src-list="[getImagePreviewUrl(result)]"
                      lazy>
                      <template #error>
                        <div class="image-slot">
                          <el-icon><Picture /></el-icon>
                        </div>
                      </template>
                    </el-image>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          <el-empty v-else-if="faceSearched" description="未找到匹配的人脸"></el-empty>
        </el-card>
      </div>

      <div v-if="activeTab === 'files'" class="files">
        <el-card class="file-list-card">
          <template #header>
            <div class="card-header">
              <span>文件列表</span>
              <div class="header-actions">
                <el-button-group class="view-toggle">
                  <el-button :type="fileViewMode === 'grid' ? 'primary' : ''" size="small" @click="fileViewMode = 'grid'">
                    <el-icon><Grid /></el-icon>
                  </el-button>
                  <el-button :type="fileViewMode === 'list' ? 'primary' : ''" size="small" @click="fileViewMode = 'list'">
                    <el-icon><List /></el-icon>
                  </el-button>
                </el-button-group>
                <el-input v-model="fileFilter.keyword" placeholder="搜索文件名" style="width: 200px; margin-right: 10px;" clearable>
                </el-input>
                <el-button type="primary" size="small" @click="refreshFileList">搜索</el-button>
                <el-button size="small" @click="refreshFileList" style="margin-left: 10px;">刷新</el-button>
              </div>
            </div>
          </template>
          
          <!-- 缩略图网格视图 -->
          <div v-if="fileViewMode === 'grid'" v-loading="loadingFiles" class="file-grid">
            <el-row :gutter="16">
              <el-col :span="6" v-for="file in fileListData" :key="file.id">
                <el-card class="file-card" >
                  <div class="file-thumbnail">
                    <el-image 
                      v-if="file.type === 'image'" 
                      :src="getFilePreviewUrlById(file.id)" 
                      fit="container" 
                      :preview-src-list="[getFilePreviewUrlById(file.id)]"
                      @click.stop>
                      <template #error>
                        <div class="file-icon-wrapper">
                          <span class="file-icon-large">🖼️</span>
                        </div>
                      </template>
                    </el-image>
                    <div v-else class="file-icon-wrapper">
                      <span class="file-icon-large">{{ getFileIcon(file.type) }}</span>
                    </div>
                  </div>
                  <div class="file-info">
                    <span class="file-name" :title="file.name">{{ file.name }}</span>
                    <div class="file-meta">
                      <el-tag :type="getTypeTagType(file.type)" size="small">{{ getTypeLabel(file.type) }}</el-tag>
                      <span class="file-size">{{ formatFileSize(file.size) }}</span>
                    </div>
                  </div>
                  <div class="file-actions">
                    <el-button size="mini" @click.stop="viewFileDetail(file)">详情</el-button>
                    <el-button size="mini" type="danger" @click.stop="deleteFile(file.id)">删除</el-button>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
          
          <!-- 列表视图 -->
          <el-table v-else :data="fileListData" style="width: 100%" v-loading="loadingFiles">
            <el-table-column prop="name" label="文件名" min-width="250">
              <template #default="scope">
                <div class="file-name-cell">
                  <span class="file-icon">{{ getFileIcon(scope.row.type) }}</span>
                  <span class="file-name-text" :title="scope.row.name">{{ scope.row.name }}</span>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="type" label="类型" width="100">
              <template #default="scope">
                <el-tag :type="getTypeTagType(scope.row.type)" size="small">{{ getTypeLabel(scope.row.type) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="subformat" label="格式" width="80"></el-table-column>
            <el-table-column prop="rid" label="区域ID" width="100"></el-table-column>
            <el-table-column prop="gid" label="分组ID" width="80"></el-table-column>
            <el-table-column prop="size" label="大小" width="100">
              <template #default="scope">
                {{ formatFileSize(scope.row.size) }}
              </template>
            </el-table-column>
            <el-table-column prop="imported_at" label="导入时间" width="160">
              <template #default="scope">
                {{ formatDate(scope.row.imported_at) }}
              </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="90">
              <template #default="scope">
                <el-tag :type="scope.row.status === 'completed' ? 'success' : 'warning'" size="small">{{ scope.row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="180" fixed="right">
              <template #default="scope">
                <el-button size="small" @click="viewFileDetail(scope.row)">详情</el-button>
                <el-button size="small" type="danger" @click="deleteFile(scope.row.id)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
          
          <div class="pagination-wrapper">
            <el-pagination
              v-model:current-page="filePagination.page"
              v-model:page-size="filePagination.pageSize"
              :page-sizes="[10, 20, 50, 100]"
              :total="filePagination.total"
              layout="total, sizes, prev, pager, next, jumper"
              @size-change="handlePageSizeChange"
              @current-change="handlePageChange">
            </el-pagination>
          </div>
        </el-card>
        
        <!-- 文件详情对话框 -->
        <el-dialog v-model="fileDetailDialogVisible" title="文件详情" width="600px">
          <div v-if="currentFileDetail" class="file-detail">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="文件ID">{{ currentFileDetail.id }}</el-descriptions-item>
              <el-descriptions-item label="文件名">{{ currentFileDetail.name }}</el-descriptions-item>
              <el-descriptions-item label="类型">
                <el-tag :type="getTypeTagType(currentFileDetail.type)">{{ getTypeLabel(currentFileDetail.type) }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="格式">{{ currentFileDetail.subformat }}</el-descriptions-item>
              <el-descriptions-item label="区域ID">{{ currentFileDetail.rid || '-' }}</el-descriptions-item>
              <el-descriptions-item label="分组ID">{{ currentFileDetail.gid !== null ? currentFileDetail.gid : '-' }}</el-descriptions-item>
              <el-descriptions-item label="文件大小">{{ formatFileSize(currentFileDetail.size) }}</el-descriptions-item>
              <el-descriptions-item label="状态">
                <el-tag :type="currentFileDetail.status === 'completed' ? 'success' : 'warning'">{{ currentFileDetail.status }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="导入时间" :span="2">
                {{ formatDate(currentFileDetail.imported_at) }}
              </el-descriptions-item>
              <el-descriptions-item label="分类" :span="2">{{ currentFileDetail.category || '-' }}</el-descriptions-item>
            </el-descriptions>
            
            <div v-if="currentFileDetail.type === 'image'" class="file-preview-section">
              <h4>预览</h4>
              <el-image 
                :src="getFilePreviewUrlById(currentFileDetail.id)" 
                fit="contain" 
                style="width: 100%; max-height: 400px;"
                :preview-src-list="[getFilePreviewUrlById(currentFileDetail.id)]">
              </el-image>
            </div>
          </div>
          <template #footer>
            <el-button @click="fileDetailDialogVisible = false">关闭</el-button>
          </template>
        </el-dialog>
      </div>
    </el-main>
  </div>
</template>

<script>
import { apiClient, config } from './api.js'
import { Plus, Picture, Grid, List } from '@element-plus/icons-vue'

export default {
  name: 'App',
  components: { Plus, Picture, Grid, List },
  data() {
    return {
      activeTab: 'home',
      fileListData: [],
      loadingFiles: false,
      fileViewMode: 'grid',
      fileFilter: {
        keyword: ''
      },
      filePagination: {
        page: 1,
        pageSize: 20,
        total: 0
      },
      fileDetailDialogVisible: false,
      currentFileDetail: null,
      
      importForm: {
        directory: '',
        rid: '',
        gid: null
      },
      importing: false,
      currentTaskId: null,
      taskPollingTimer: null,
      tasks: [],
      importProgress: {
        show: false,
        percentage: 0,
        message: '',
        status: ''
      },
      
      semanticSearchForm: {
        query: '',
        fileType: ''
      },
      searching: false,
      semanticResults: [],
      semanticSearched: false,
      
      imageSearchForm: {
        imageUrl: '',
        imageFile: null,
        gid: null
      },
      imageSearching: false,
      imageSearchResults: [],
      imageSearched: false,
      
      faceSearchForm: {
        imageUrl: '',
        imageFile: null,
        gid: null
      },
      faceSearching: false,
      faceSearchResults: [],
      faceSearched: false
    }
  },
  mounted() {
    this.refreshFileList()
    this.refreshTasks()
  },
  methods: {
    handleTabChange(tab) {
      this.activeTab = tab
    },
    
    async startBatchImport() {
      if (!this.importForm.directory) {
        this.$message.warning('请输入目录路径')
        return
      }
      
      this.importing = true
      this.importProgress.show = true
      this.importProgress.percentage = 0
      this.importProgress.message = '正在导入...'
      this.importProgress.status = ''
      this.currentTaskId = null
      
      try {
        const formData = new FormData()
        formData.append('directory', this.importForm.directory)
        if (this.importForm.rid) formData.append('rid', this.importForm.rid)
        if (this.importForm.gid !== null) formData.append('gid', this.importForm.gid)
        
        const response = await apiClient.post('/file/batch_import', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        
        this.importProgress.percentage = 100
        this.importProgress.status = 'success'
        this.importProgress.message = `导入完成！成功: ${response.data.success_count}, 失败: ${response.data.failed_count}`
        
        if (response.data.failed_files && response.data.failed_files.length > 0) {
          console.error('失败文件:', response.data.failed_files)
        }
        
        this.$message.success(this.importProgress.message)
        this.refreshFileList()
        
      } catch (error) {
        this.importProgress.status = 'exception'
        this.$message.error('批量导入失败: ' + (error.response?.data?.detail || error.message))
        this.importProgress.message = '导入失败'
      } finally {
        this.importing = false
      }
    },
    
    async startBatchImportAsync() {
      if (!this.importForm.directory) {
        this.$message.warning('请输入目录路径')
        return
      }
      
      this.importing = true
      this.importProgress.show = true
      this.importProgress.percentage = 0
      this.importProgress.message = '正在提交任务...'
      this.importProgress.status = ''
      
      try {
        const formData = new FormData()
        formData.append('directory', this.importForm.directory)
        if (this.importForm.rid) formData.append('rid', this.importForm.rid)
        if (this.importForm.gid !== null) formData.append('gid', this.importForm.gid)
        
        const response = await apiClient.post('/file/batch_import_async', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        
        this.currentTaskId = response.data.task_id
        this.importProgress.message = '任务已提交，后台处理中...'
        this.$message.success(response.data.message)
        
        this.startTaskPolling(response.data.task_id)
        this.refreshTasks()
        
      } catch (error) {
        this.importProgress.status = 'exception'
        this.$message.error('提交任务失败: ' + (error.response?.data?.detail || error.message))
        this.importProgress.message = '提交失败'
        this.importing = false
      }
    },
    
    startTaskPolling(taskId) {
      if (this.taskPollingTimer) {
        clearInterval(this.taskPollingTimer)
      }
      
      this.taskPollingTimer = setInterval(async () => {
        try {
          const response = await apiClient.get(`/task/${taskId}`)
          const task = response.data.task
          
          this.importProgress.percentage = task.progress
          this.importProgress.message = task.message
          
          if (task.status === 'completed') {
            this.importProgress.status = 'success'
            this.stopTaskPolling()
            this.importing = false
            this.$message.success(task.result?.message || '任务完成')
            this.refreshFileList()
            this.refreshTasks()
          } else if (task.status === 'failed') {
            this.importProgress.status = 'exception'
            this.stopTaskPolling()
            this.importing = false
            this.$message.error('任务失败: ' + task.error)
            this.refreshTasks()
          }
          
        } catch (error) {
          console.error('轮询任务状态失败:', error)
        }
      }, 2000)
    },
    
    stopTaskPolling() {
      if (this.taskPollingTimer) {
        clearInterval(this.taskPollingTimer)
        this.taskPollingTimer = null
      }
    },
    
    async refreshTasks() {
      try {
        const response = await apiClient.get('/tasks')
        this.tasks = response.data.tasks
      } catch (error) {
        console.error('获取任务列表失败:', error)
      }
    },
    
    viewTaskDetail(task) {
      let detail = `任务ID: ${task.id}\n`
      detail += `类型: ${task.type}\n`
      detail += `状态: ${this.getTaskStatusText(task.status)}\n`
      detail += `进度: ${task.progress}%\n`
      detail += `消息: ${task.message}\n`
      
      if (task.result) {
        detail += `\n结果:\n${JSON.stringify(task.result, null, 2)}`
      }
      
      if (task.error) {
        detail += `\n错误: ${task.error}`
      }
      
      alert(detail)
    },
    
    getTaskStatusType(status) {
      const map = {
        'pending': 'info',
        'running': 'primary',
        'completed': 'success',
        'failed': 'danger'
      }
      return map[status] || 'info'
    },
    
    getTaskStatusText(status) {
      const map = {
        'pending': '等待中',
        'running': '运行中',
        'completed': '已完成',
        'failed': '失败'
      }
      return map[status] || status
    },
    
    async refreshFileList() {
      this.loadingFiles = true
      try {
        const skip = (this.filePagination.page - 1) * this.filePagination.pageSize
        const params = {
          skip: skip,
          limit: this.filePagination.pageSize
        }
        if (this.fileFilter.keyword) {
          params.keyword = this.fileFilter.keyword
        }
        
        const response = await apiClient.get('/file/list', { params })
        this.fileListData = response.data.items
        this.filePagination.total = response.data.total
      } catch (error) {
        this.$message.error('获取文件列表失败')
      } finally {
        this.loadingFiles = false
      }
    },
    
    handlePageChange(page) {
      this.filePagination.page = page
      this.refreshFileList()
    },
    
    handlePageSizeChange(pageSize) {
      this.filePagination.pageSize = pageSize
      this.filePagination.page = 1
      this.refreshFileList()
    },
    
    async viewFileDetail(file) {
      try {
        const response = await apiClient.get(`/file/${file.id}`)
        this.currentFileDetail = response.data
        this.fileDetailDialogVisible = true
      } catch (error) {
        this.$message.error('获取文件详情失败')
      }
    },
    
    getFileIcon(type) {
      const icons = {
        'document': '📄',
        'image': '🖼️',
        'video': '🎬'
      }
      return icons[type] || '📁'
    },
    
    formatDate(dateStr) {
      if (!dateStr) return '-'
      const d = new Date(dateStr)
      return d.toLocaleString('zh-CN')
    },
    
    getFilePreviewUrlById(fileId) {
      return `${config.publicBaseUrl}/preview/${fileId}`
    },
    
    async deleteFile(fileId) {
      try {
        await this.$confirm('确定要删除这个文件吗？', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
        await apiClient.delete(`/file/${fileId}`)
        this.$message.success('文件删除成功')
        this.refreshFileList()
      } catch (error) {
        if (error !== 'cancel') {
          this.$message.error('文件删除失败')
        }
      }
    },
    
    async doSemanticSearch() {
      if (!this.semanticSearchForm.query.trim()) {
        this.$message.warning('请输入搜索内容')
        return
      }
      
      this.searching = true
      this.semanticSearched = false
      
      try {
        const response = await apiClient.post('/search', null, {
          params: {
            query: this.semanticSearchForm.query,
            file_type: this.semanticSearchForm.fileType
          }
        })
        this.semanticResults = response.data
        this.semanticSearched = true
      } catch (error) {
        this.$message.error('搜索失败: ' + (error.response?.data?.detail || error.message))
      } finally {
        this.searching = false
      }
    },
    
    handleImageChange(file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        this.imageSearchForm.imageUrl = e.target.result
        this.imageSearchForm.imageFile = file.raw
      }
      reader.readAsDataURL(file.raw)
    },
    
    async doImageSearch() {
      if (!this.imageSearchForm.imageFile) {
        this.$message.warning('请先上传图片')
        return
      }
      
      this.imageSearching = true
      this.imageSearched = false
      
      try {
        const formData = new FormData()
        formData.append('file', this.imageSearchForm.imageFile)
        if (this.imageSearchForm.gid !== null) formData.append('gid', this.imageSearchForm.gid)
        
        const response = await apiClient.post('/search/image', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        this.imageSearchResults = response.data
        this.imageSearched = true
      } catch (error) {
        this.$message.error('搜索失败: ' + (error.response?.data?.detail || error.message))
      } finally {
        this.imageSearching = false
      }
    },
    
    handleFaceImageChange(file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        this.faceSearchForm.imageUrl = e.target.result
        this.faceSearchForm.imageFile = file.raw
      }
      reader.readAsDataURL(file.raw)
    },
    
    async doFaceSearch() {
      if (!this.faceSearchForm.imageFile) {
        this.$message.warning('请先上传图片')
        return
      }
      
      this.faceSearching = true
      this.faceSearched = false
      
      try {
        const formData = new FormData()
        formData.append('file', this.faceSearchForm.imageFile)
        if (this.faceSearchForm.gid !== null) formData.append('gid', this.faceSearchForm.gid)
        
        const response = await apiClient.post('/search/face', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
        this.faceSearchResults = response.data
        this.faceSearched = true
      } catch (error) {
        this.$message.error('搜索失败: ' + (error.response?.data?.detail || error.message))
      } finally {
        this.faceSearching = false
      }
    },
    
    getTypeLabel(type) {
      const labels = {
        'document': '文档',
        'image': '图片',
        'video': '视频'
      }
      return labels[type] || type
    },
    
    getTypeTagType(type) {
      const types = {
        'document': '',
        'image': 'success',
        'video': 'warning'
      }
      return types[type] || ''
    },
    
    getImagePreviewUrl(result) {
      if (result.frame_path) {
        return `${config.publicBaseUrl}/file?path=${encodeURIComponent(result.frame_path)}`
      }
      if (result.file_info) {
        return `${config.publicBaseUrl}/preview/${result.file_info.id}`
      }
      return ''
    },
    
    formatFileSize(size) {
      if (size < 1024) return size + ' B'
      if (size < 1024 * 1024) return (size / 1024).toFixed(2) + ' KB'
      if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed(2) + ' MB'
      return (size / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
    }
  }
}
</script>

<style scoped>
body{
  padding:0;
  margin:0
}
.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background-color: #409EFF;
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}
.header .el-menu.el-menu--horizontal {
  min-width:615px;
}

.header h1 {
  margin: 0;
  font-size: 20px;
}

.main {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.view-toggle {
  margin-right: 15px;
}

.file-grid {
  padding: 10px 0;
}

.file-card {
  height: 310px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.file-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.file-thumbnail {
  height: 150px;
  overflow: hidden;
  margin-bottom: 12px;
  background: #f5f7fa;
  border-radius: 4px;
}

.file-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.file-icon-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.file-icon-large {
  font-size: 48px;
}

.file-info {
  padding: 0 8px;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 12px;
  color: #999;
}

.file-size {
  font-size: 12px;
  color: #999;
}

.file-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #eee;
}

.header-actions {
  display: flex;
  align-items: center;
}

.file-name-cell {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-icon {
  font-size: 18px;
}

.file-name-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.file-detail {
  padding: 10px 0;
}

.file-preview-section {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.file-preview-section h4 {
  margin: 0 0 15px 0;
  color: #666;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.view-toggle {
  margin-right: 15px;
}

.file-grid {
  padding: 10px 0;
}

.file-card {
  height: 310px;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.file-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.file-thumbnail {
  height: 150px;
  overflow: hidden;
  margin-bottom: 12px;
  background: #f5f7fa;
  border-radius: 4px;
}

.file-thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.file-icon-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.file-icon-large {
  font-size: 48px;
}

.file-info {
  padding: 0 8px;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 12px;
  color: #999;
}

.file-size {
  font-size: 12px;
  color: #999;
}

.file-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #eee;
}

.welcome-content {
  padding: 20px 0;
}

.feature-card {
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.feature-icon {
  font-size: 48px;
  margin-bottom: 10px;
}

.feature-card h4 {
  margin: 10px 0 5px 0;
}

.feature-card p {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.import-progress {
  margin-top: 20px;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.search-form {
  margin-bottom: 20px;
}

.search-results {
  margin-top: 20px;
}

.search-results h3 {
  margin-bottom: 20px;
}

.comparison-section {
  margin-bottom: 30px;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.comparison-section h4 {
  margin-top: 0;
  margin-bottom: 15px;
}

.result-card {
  margin-bottom: 20px;
}

.result-type {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.result-card h4 {
  margin: 0 0 10px 0;
  font-size: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-content {
  color: #666;
  font-size: 14px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  margin-bottom: 10px;
}

.result-image {
  margin-top: 10px;
}

.image-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 150px;
  background: #f5f7fa;
  color: #909399;
}

.image-uploader {
  width: 200px;
  height: 200px;
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s;
}

.image-uploader:hover {
  border-color: #409EFF;
}

.uploaded-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-uploader-icon {
  font-size: 48px;
  color: #8c939d;
  width: 200px;
  height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
