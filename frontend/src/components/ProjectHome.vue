<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'
import { api } from '../api'
import { locale, tr } from '../i18n'
import type { HealthResponse, PhotoProject } from '../types'
import LanguageToggle from './LanguageToggle.vue'

defineProps<{ projects: PhotoProject[]; health: HealthResponse | null }>()
const emit = defineEmits<{ open: [project: PhotoProject]; created: [project: PhotoProject] }>()

const fileInput = ref<HTMLInputElement | null>(null)
const folderInput = ref<HTMLInputElement | null>(null)
const selectedFiles = ref<File[]>([])
const projectName = ref('')
const busy = ref(false)
const error = ref('')
const previews = ref<string[]>([])

const totalSize = computed(() => selectedFiles.value.reduce((sum, file) => sum + file.size, 0))
const totalSizeLabel = computed(() => `${(totalSize.value / 1024 / 1024).toFixed(1)} MB`)
const selectedFolder = computed(() => selectedFiles.value[0]?.webkitRelativePath.split('/')[0] || '')

function clearPreviews() {
  previews.value.forEach(URL.revokeObjectURL)
  previews.value = []
}

function selectFiles(event: Event) {
  const input = event.target as HTMLInputElement
  const values = [...(input.files || [])]
  if (!values.length) return
  clearPreviews()
  selectedFiles.value = values
  previews.value = values.slice(0, 5).map(URL.createObjectURL)
  if (!projectName.value) {
    projectName.value = values[0].webkitRelativePath.split('/')[0] || `${tr('Photo project', '照片项目')} ${new Date().toLocaleDateString()}`
  }
}

async function createProject() {
  if (!selectedFiles.value.length || !projectName.value.trim()) return
  busy.value = true
  error.value = ''
  try {
    const project = await api.createProject(projectName.value)
    const result = await api.uploadFiles(project.id, selectedFiles.value)
    emit('created', result.project)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : tr('Could not create the project', '无法创建项目')
  } finally {
    busy.value = false
  }
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(locale.value === 'zh' ? 'zh-CN' : 'en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date(value))
}

onBeforeUnmount(clearPreviews)
</script>

<template>
  <div class="home-shell">
    <header class="home-header page-width">
      <button class="wordmark" type="button"><span>✦</span> Memora</button>
      <nav><a href="#workflow">{{ tr('Workflow', '工作流程') }}</a><a href="#capabilities">{{ tr('Capabilities', '功能') }}</a><a href="#projects">{{ tr('Projects', '项目') }}</a></nav>
      <div class="header-tools"><LanguageToggle /><div class="api-state"><i :class="{ online: health?.status === 'ok' }"></i>{{ health?.status === 'ok' ? tr('AI service connected', 'AI 服务已连接') : tr('Waiting for FastAPI', '等待 FastAPI') }}</div></div>
    </header>

    <main>
      <section class="hero page-width">
        <div class="hero-message">
          <p class="section-label">PERSONAL PHOTO INTELLIGENCE</p>
          <h1 v-if="locale === 'en'">Turn scattered photos<br />into <span>searchable memories.</span></h1><h1 v-else>把散落的照片，<br />整理成一段段<span>可以寻找的记忆。</span></h1>
          <p class="hero-lead">{{ tr('Create a photo project from a local folder. Memora gives every project its own space for semantic search, people clustering, event discovery, duplicate cleanup and best-shot selection.', '创建一个照片项目，读取本地文件夹。Memora 会在独立空间中完成语义检索、人物聚类、事件发现、相似照片整理和最佳照片推荐。') }}</p>
          <div class="trust-row"><span>{{ tr('Local first', '本地优先') }}</span><span>{{ tr('Isolated projects', '项目隔离') }}</span><span>{{ tr('Exportable results', '可导出结果') }}</span></div>
        </div>

        <div class="import-panel">
          <div class="panel-heading"><div><p class="section-label">NEW PROJECT</p><h2>{{ tr('Start organizing photos', '开始整理照片') }}</h2></div><span class="step-tag">01 / IMPORT</span></div>
          <div v-if="!selectedFiles.length" class="drop-zone" @click="folderInput?.click()">
            <div class="upload-symbol">↥</div><h3>{{ tr('Read a photo folder', '读取一个照片文件夹') }}</h3><p>{{ tr('Keep the original folder structure. Supports JPG, PNG, WEBP and TIFF.', '保留原有子文件夹结构，支持 JPG、PNG、WEBP 和 TIFF') }}</p>
            <div class="import-actions"><button class="primary-action" type="button" @click.stop="folderInput?.click()">{{ tr('Choose folder', '选择文件夹') }}</button><button class="quiet-action" type="button" @click.stop="fileInput?.click()">{{ tr('Choose photos', '选择照片') }}</button></div>
          </div>
          <div v-else class="selection-review">
            <div class="preview-stack"><img v-for="preview in previews" :key="preview" :src="preview" alt="Selected photo preview" /></div>
            <div class="selection-title"><div><strong>{{ selectedFolder || tr('Selected photos', '已选择照片') }}</strong><span>{{ selectedFiles.length }} {{ tr('photos', '张照片') }} · {{ totalSizeLabel }}</span></div><button type="button" @click="selectedFiles = []; clearPreviews()">{{ tr('Choose again', '重新选择') }}</button></div>
            <label>{{ tr('Project name', '项目名称') }}<input v-model="projectName" maxlength="120" :placeholder="tr('Example: Summer 2025', '例如：2025 夏季旅行')" /></label>
            <button class="primary-action full" type="button" :disabled="busy || !projectName.trim()" @click="createProject">{{ busy ? tr('Uploading and creating project…', '正在上传并创建项目…') : tr('Create project and open workspace', '创建项目并进入工作台') }} <span>→</span></button>
          </div>
          <p v-if="error" class="inline-error">{{ error }}</p>
          <input ref="folderInput" class="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.webp,.tif,.tiff" multiple webkitdirectory @change="selectFiles" />
          <input ref="fileInput" class="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.webp,.tif,.tiff" multiple @change="selectFiles" />
        </div>
      </section>

      <section id="workflow" class="workflow-strip"><div class="page-width workflow-inner"><p class="section-label">FROM FOLDER TO STORY</p><ol><li><span>01</span><div><strong>{{ tr('Import', '导入') }}</strong><small>{{ tr('Read local files and folders', '读取本地文件与文件夹') }}</small></div></li><li><span>02</span><div><strong>{{ tr('Analyze', '分析') }}</strong><small>{{ tr('Extract visuals, people and metadata', '提取视觉、人物和元数据') }}</small></div></li><li><span>03</span><div><strong>{{ tr('Organize', '整理') }}</strong><small>{{ tr('Search, cluster and select', '搜索、聚类与最佳照片') }}</small></div></li><li><span>04</span><div><strong>{{ tr('Export', '导出') }}</strong><small>{{ tr('Albums, indexes and reports', '相册、清单与分析报告') }}</small></div></li></ol></div></section>

      <section id="capabilities" class="capabilities page-width">
        <div class="section-heading"><div><p class="section-label">WHAT MEMORA DOES</p><h2>{{ tr('A complete photo organization workflow', '一套完整的照片整理工作流') }}</h2></div><p>{{ tr('Every capability works on the same photo project, so results can be combined instead of living as isolated algorithm demos.', '所有功能都围绕同一个照片项目运行，结果可以互相组合，而不是一组彼此孤立的算法演示。') }}</p></div>
        <div class="feature-grid">
          <article><span>⌕</span><p>01 · RETRIEVAL</p><h3>{{ tr('Natural-language search', '自然语言搜索') }}</h3><div>{{ tr('Search with bilingual descriptions, dates and locations.', '中英文描述、时间和地点条件共同检索照片。') }}</div></article>
          <article><span>◎</span><p>02 · PEOPLE</p><h3>{{ tr('Automatic people groups', '人物自动分组') }}</h3><div>{{ tr('InsightFace clustering with names, merges and human corrections.', 'InsightFace 聚类，并支持命名、合并和人工修正。') }}</div></article>
          <article><span>◷</span><p>03 · EVENTS</p><h3>{{ tr('Events and journeys', '事件与旅程发现') }}</h3><div>{{ tr('Reconstruct events from time, visual similarity and GPS.', '结合拍摄时间、视觉相似度与 GPS 重建事件。') }}</div></article>
          <article><span>◫</span><p>04 · CURATION</p><h3>{{ tr('Similar and best shots', '相似照片与最佳照片') }}</h3><div>{{ tr('Find bursts and duplicates, then keep the clearest version.', '识别连拍和重复内容，挑选更清晰、更自然的版本。') }}</div></article>
          <article class="export-feature"><span>⇩</span><p>05 · EXPORT</p><h3>{{ tr('Take your results with you', '整理结果可带走') }}</h3><div>{{ tr('Export CSV indexes, JSON manifests and curated photo packages.', '导出照片索引 CSV、项目分析清单 JSON，并可生成精选相册。') }}</div></article>
        </div>
      </section>

      <section id="projects" class="recent-projects page-width">
        <div class="section-heading"><div><p class="section-label">YOUR LIBRARY</p><h2>{{ tr('Recent projects', '最近项目') }}</h2></div><button class="quiet-action" type="button" @click="folderInput?.click()">＋ {{ tr('New project', '新建项目') }}</button></div>
        <div v-if="projects.length" class="project-list"><button v-for="project in projects" :key="project.id" type="button" @click="emit('open', project)"><div class="project-cover"><span>{{ project.photo_count }}</span><small>PHOTOS</small></div><div class="project-copy"><span :class="['status-dot', project.status]"></span><strong>{{ project.name }}</strong><small>{{ formatDate(project.updated_at) }} · {{ project.analyzed_count ? `${project.analyzed_count} ${tr('analyzed', '已分析')}` : tr('Waiting for analysis', '等待分析') }}</small></div><b>→</b></button></div>
        <div v-else class="empty-projects"><p>{{ tr('No photo projects yet. Choose a folder above to begin.', '还没有照片项目。选择上方的文件夹开始。') }}</p></div>
      </section>
    </main>
    <footer class="page-width"><span>MEMORA AI · PRIVATE PHOTO WORKSPACE</span><span>FastAPI {{ health?.status === 'ok' ? 'ONLINE' : 'OFFLINE' }}</span></footer>
  </div>
</template>
