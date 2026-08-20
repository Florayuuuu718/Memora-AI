<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, exportUrl, mediaUrl } from '../api'
import { locale, tr } from '../i18n'
import type { BestShot, EventItem, HealthResponse, JourneyItem, PersonGroup, PhotoProject, ProjectPhoto, SearchResult, SimilarGroup } from '../types'
import LanguageToggle from './LanguageToggle.vue'

type WorkspaceTab = 'overview' | 'search' | 'people' | 'events' | 'similar' | 'best' | 'export'

const props = defineProps<{ project: PhotoProject; health: HealthResponse | null }>()
const emit = defineEmits<{ back: []; updated: [project: PhotoProject] }>()
const tab = ref<WorkspaceTab>('overview')
const project = ref(props.project)
const photos = ref<ProjectPhoto[]>([])
const searchResults = ref<SearchResult[]>([])
const events = ref<EventItem[]>([])
const groups = ref<SimilarGroup[]>([])
const bestShots = ref<BestShot[]>([])
const people = ref<PersonGroup[]>([])
const query = ref('')
const busy = ref(false)
const error = ref('')
const addFilesInput = ref<HTMLInputElement | null>(null)
const journeys = ref<JourneyItem[]>([])
const selectedPerson = ref<PersonGroup | null>(null)
const selectedEvent = ref<EventItem | null>(null)
const selectedJourney = ref<JourneyItem | null>(null)
const editName = ref('')
const editNote = ref('')
const llmGenerating = ref<'name' | 'note' | null>(null)
const loadedTabs = new Set<WorkspaceTab>()

const photoById = computed(() => new Map(photos.value.map((photo) => [photo.id, photo])))
const analyzedPercent = computed(() => project.value.photo_count ? Math.round(project.value.analyzed_count / project.value.photo_count * 100) : 0)
const folders = computed(() => new Set(photos.value.map((photo) => photo.relative_path.split('/').slice(0, -1).join('/') || 'Root')).size)

async function loadPhotos() { photos.value = (await api.photos(project.value.id)).photos }

function clearDerivedResults() {
  searchResults.value = []
  events.value = []
  journeys.value = []
  groups.value = []
  bestShots.value = []
  people.value = []
  loadedTabs.clear()
}

async function analyze() {
  busy.value = true
  error.value = ''
  project.value = { ...project.value, status: 'analyzing' }
  try {
    project.value = (await api.analyze(project.value.id)).project
    clearDerivedResults()
    emit('updated', project.value)
    await loadPhotos()
  } catch (cause) {
    project.value = { ...project.value, status: 'uploaded' }
    error.value = cause instanceof Error ? cause.message : tr('Analysis failed', '分析失败')
  } finally { busy.value = false }
}

async function uploadMore(event: Event) {
  const input = event.target as HTMLInputElement
  const files = [...(input.files || [])]
  if (!files.length) return
  busy.value = true
  try {
    project.value = (await api.uploadFiles(project.value.id, files)).project
    clearDerivedResults()
    emit('updated', project.value)
    await loadPhotos()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : tr('Upload failed', '上传失败')
  } finally {
    busy.value = false
    input.value = ''
  }
}

async function runSearch() {
  if (!query.value.trim() || project.value.encoder !== 'open_clip') return
  busy.value = true
  error.value = ''
  try { searchResults.value = (await api.search(project.value.id, query.value.trim())).results }
  catch (cause) { error.value = cause instanceof Error ? cause.message : tr('Search failed', '搜索失败') }
  finally { busy.value = false }
}

async function selectTab(next: WorkspaceTab) {
  tab.value = next
  error.value = ''
  if (project.value.status !== 'ready' || busy.value) return
  if (!['events', 'similar', 'best', 'people'].includes(next) || loadedTabs.has(next)) return
  busy.value = true
  try {
    if (next === 'events') {
      const [eventResponse, journeyResponse, peopleResponse] = await Promise.all([
        api.events(project.value.id),
        api.journeys(project.value.id),
        api.people(project.value.id),
      ])
      events.value = eventResponse.events
      journeys.value = journeyResponse.journeys
      people.value = peopleResponse.groups
      loadedTabs.add('people')
    }
    if (next === 'similar') groups.value = (await api.similar(project.value.id)).groups
    if (next === 'best') bestShots.value = (await api.bestShots(project.value.id)).photos
    if (next === 'people') people.value = (await api.people(project.value.id)).groups
    loadedTabs.add(next)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : tr('Could not load analysis results', '无法加载分析结果')
  } finally { busy.value = false }
}

async function analyzePeople() {
  busy.value = true
  error.value = ''
  try {
    people.value = (await api.clusterPeople(project.value.id)).groups
    events.value = []
    journeys.value = []
    groups.value = []
    bestShots.value = []
    loadedTabs.delete('events')
    loadedTabs.delete('similar')
    loadedTabs.delete('best')
    loadedTabs.add('people')
  }
  catch (cause) { error.value = cause instanceof Error ? cause.message : tr('People clustering failed', '人物聚类失败') }
  finally { busy.value = false }
}

function resultUrl(result: SearchResult) {
  const indexedPhoto = photoById.value.get(result.photo_id)
  return mediaUrl(indexedPhoto?.url || result.thumbnail_url)
}

function dateLabel(value: string | null) {
  if (!value) return tr('Unknown date', '日期未知')
  return new Intl.DateTimeFormat(locale.value === 'zh' ? 'zh-CN' : 'en-US', { dateStyle: 'medium' }).format(new Date(value))
}

function eventLabel(event: EventItem) {
  if (event.name) return event.name
  const names = (event.person_ids || []).map((id) => people.value.find((person) => person.id === id)?.name).filter(Boolean)
  return names.length ? names.join(' · ') : tr(`Event ${event.id + 1}`, `事件 ${event.id + 1}`)
}

function openPerson(person: PersonGroup) {
  selectedPerson.value = person; selectedEvent.value = null; selectedJourney.value = null
  editName.value = person.name || ''; editNote.value = person.note || ''
}
function openEvent(event: EventItem) {
  selectedEvent.value = event; selectedPerson.value = null; selectedJourney.value = null
  editName.value = event.name || ''; editNote.value = event.note || ''
}
function openJourney(journey: JourneyItem) {
  selectedJourney.value = journey; selectedPerson.value = null; selectedEvent.value = null
  editName.value = journey.name || ''; editNote.value = journey.note || ''
}
async function saveDetail() {
  const item = selectedPerson.value || selectedEvent.value || selectedJourney.value
  if (!item) return
  const kind = selectedPerson.value ? 'people' : selectedEvent.value ? 'events' : 'journeys'
  busy.value = true
  try {
    const response = await api.saveAnnotation(project.value.id, kind, item.id, { name: editName.value, note: editNote.value })
    Object.assign(item, response.annotation)
  } catch (cause) { error.value = cause instanceof Error ? cause.message : tr('Could not save details', '无法保存详情') }
  finally { busy.value = false }
}

async function generateDetail(field: 'name' | 'note') {
  const item = selectedPerson.value || selectedEvent.value || selectedJourney.value
  if (!item) return
  const kind = selectedPerson.value ? 'people' : selectedEvent.value ? 'events' : 'journeys'
  llmGenerating.value = field
  error.value = ''
  try {
    const response = await api.generateAnnotation(project.value.id, kind, item.id, field)
    if (field === 'name') editName.value = response.value
    else editNote.value = response.value
  } catch (cause) { error.value = cause instanceof Error ? cause.message : tr('LLM generation failed', 'LLM 生成失败') }
  finally { llmGenerating.value = null }
}

onMounted(loadPhotos)
</script>

<template>
  <div class="workspace-shell">
    <aside class="workspace-sidebar">
      <button class="wordmark workspace-brand" type="button" @click="emit('back')"><span>✦</span> Memora</button>
      <button class="back-link" type="button" @click="emit('back')">← {{ tr('Back to project dashboard', '返回项目控制台') }}</button>
      <div class="project-identity"><div class="folder-mark">▰</div><div><strong>{{ project.name }}</strong><span>{{ project.photo_count }} {{ tr('photos', '张照片') }}</span></div></div>
      <nav class="workspace-nav">
        <p>WORKSPACE</p>
        <button :class="{ active: tab === 'overview' }" @click="selectTab('overview')"><span>▦</span>{{ tr('Overview', '项目总览') }}</button>
        <button :class="{ active: tab === 'search' }" @click="selectTab('search')"><span>⌕</span>{{ tr('AI Search', 'AI 搜索') }}</button>
        <button :class="{ active: tab === 'people' }" @click="selectTab('people')"><span>●</span>{{ tr('People', '人物') }}</button>
        <button :class="{ active: tab === 'events' }" @click="selectTab('events')"><span>◆</span>{{ tr('Events & Trips', '事件与旅程') }}</button>
        <button :class="{ active: tab === 'similar' }" @click="selectTab('similar')"><span>◫</span>{{ tr('Similar Photos', '相似照片') }}</button>
        <button :class="{ active: tab === 'best' }" @click="selectTab('best')"><span>◇</span>{{ tr('Best Shots', '最佳照片') }}</button>
        <p>OUTPUT</p>
        <button :class="{ active: tab === 'export' }" @click="selectTab('export')"><span>⇩</span>{{ tr('Export Center', '导出中心') }}</button>
      </nav>
      <div class="sidebar-status"><div><i :class="{ online: health?.status === 'ok' }"></i>{{ project.encoder === 'open_clip' ? 'OpenCLIP' : (project.encoder || health?.encoder || 'AI service') }}</div><span>{{ analyzedPercent }}% {{ tr('analyzed', '已分析') }}</span></div>
    </aside>

    <main class="workspace-main">
      <header class="workspace-topbar">
        <div><p class="section-label">PHOTO PROJECT / {{ project.id }}</p><h1>{{ project.name }}</h1></div>
        <div class="top-actions"><LanguageToggle /><button class="quiet-action" type="button" @click="addFilesInput?.click()">＋ {{ tr('Add photos', '添加照片') }}</button><button class="primary-action" type="button" :disabled="busy || project.status === 'ready'" @click="analyze">{{ project.status === 'ready' ? tr('✓ Analysis complete', '✓ 分析完成') : project.status === 'analyzing' ? tr('Analyzing…', '分析中…') : tr('Start AI analysis', '开始 AI 分析') }}</button></div>
      </header>
      <input ref="addFilesInput" class="visually-hidden" type="file" accept=".jpg,.jpeg,.png,.webp,.tif,.tiff" multiple @change="uploadMore" />
      <div v-if="error" class="workspace-error">{{ error }}</div>

      <section v-if="tab === 'overview'" class="workspace-content">
        <div v-if="project.status !== 'ready'" class="analysis-callout"><div><p class="section-label">NEXT STEP</p><h2>{{ tr('Let AI understand this photo project', '让 AI 理解这个照片项目') }}</h2><p>{{ tr('Analysis creates visual embeddings, reads time and GPS data, scores image quality, and builds a private index for this project.', '分析会为每张照片生成视觉向量、读取时间和 GPS、计算画质，并建立该项目的专属索引。') }}</p></div><button class="primary-action" type="button" :disabled="busy" @click="analyze">{{ busy ? tr('Analyzing, please wait…', '正在分析，请稍候…') : tr(`Analyze ${project.photo_count} photos →`, `分析 ${project.photo_count} 张照片 →`) }}</button></div>
        <div class="metric-row"><article><span>{{ tr('Photos', '照片') }}</span><strong>{{ project.photo_count }}</strong><small>{{ tr('Imported files', '已导入文件') }}</small></article><article><span>{{ tr('Folders', '文件夹') }}</span><strong>{{ folders }}</strong><small>{{ tr('Original structure preserved', '保留原始结构') }}</small></article><article><span>{{ tr('Analysis progress', '分析进度') }}</span><strong>{{ analyzedPercent }}%</strong><small>{{ project.status === 'ready' ? tr('Index ready', '索引可用') : tr('Waiting for processing', '等待处理') }}</small></article><article><span>{{ tr('Engine', '引擎') }}</span><strong class="engine-name">{{ project.encoder === 'open_clip' ? 'OpenCLIP' : (project.encoder || health?.encoder || '—') }}</strong><small>{{ tr('Project index encoder', '项目索引编码器') }}</small></article></div>
        <div class="content-heading"><div><p class="section-label">PROJECT PHOTOS</p><h2>{{ tr('Photo browser', '照片浏览') }}</h2></div><span>{{ photos.length }} ITEMS</span></div>
        <div class="asset-grid"><article v-for="photo in photos" :key="photo.id"><img :src="mediaUrl(photo.url)" :alt="photo.filename" loading="lazy" /><div><strong>{{ photo.filename }}</strong><span>{{ photo.relative_path }}</span></div></article></div>
      </section>

      <section v-else-if="tab === 'search'" class="workspace-content narrow-content">
        <div class="tool-heading"><p class="section-label">SEMANTIC RETRIEVAL</p><h2>{{ tr('Find photos with one sentence', '用一句话找到照片') }}</h2><p>{{ tr('Describe a person, scene, time, or place—for example, “photos taken at the beach last summer.”', '可以描述人物、场景、时间或地点，例如“去年夏天在海边拍的照片”。') }}</p></div>
        <div v-if="project.encoder !== 'open_clip'" class="analysis-callout"><div><p class="section-label">OPENCLIP REQUIRED</p><h2>{{ tr('Rebuild the semantic index', '重建语义索引') }}</h2><p>{{ tr('This project uses the lightweight test encoder, which cannot provide semantic text-to-image retrieval. Re-analyze it with OpenCLIP before searching.', '此项目使用的是轻量测试编码器，无法提供真正的图文语义检索。请先用 OpenCLIP 重新分析。') }}</p></div><button class="primary-action" type="button" :disabled="busy" @click="analyze">{{ busy ? tr('Analyzing…', '正在分析…') : tr('Re-analyze with OpenCLIP', '使用 OpenCLIP 重新分析') }}</button></div>
        <form class="workspace-search" @submit.prevent="runSearch"><span>⌕</span><input v-model="query" :placeholder="tr('Describe the photos you want to find…', '描述你想找的照片…')" /><button class="primary-action" :disabled="busy || project.status !== 'ready' || project.encoder !== 'open_clip'">{{ tr('Search', '搜索') }}</button></form>
        <div v-if="searchResults.length" class="content-heading"><div><p class="section-label">RESULTS</p><h2>{{ searchResults.length }} {{ tr('matches', '个匹配结果') }}</h2></div></div>
        <div class="asset-grid search-assets"><article v-for="result in searchResults" :key="result.photo_id"><img v-if="resultUrl(result)" :src="resultUrl(result)" :alt="result.photo_id" loading="lazy" /><div><strong>{{ Math.round(result.score * 100) }}% {{ tr('match', '匹配') }}</strong><span>{{ dateLabel(result.captured_at) }}</span></div></article></div>
      </section>

      <section v-else-if="tab === 'people'" class="workspace-content narrow-content">
        <template v-if="selectedPerson">
          <button class="detail-back" type="button" @click="selectedPerson = null">← {{ tr('All people', '返回所有人物') }}</button>
          <header class="record-header"><div class="record-avatar"><img v-if="selectedPerson.cover_url" :src="mediaUrl(selectedPerson.cover_url)" alt="Person cover" /><span v-else>●</span></div><div><p class="section-label">PERSON PROFILE / {{ String(selectedPerson.id + 1).padStart(2, '0') }}</p><h2>{{ selectedPerson.name || tr(`Person ${selectedPerson.id + 1}`, `人物 ${selectedPerson.id + 1}`) }}</h2><p>{{ selectedPerson.photo_ids.length }} {{ tr('photos in this person group', '张照片归属于此人物') }}</p></div></header>
          <div class="record-layout"><div class="record-gallery"><div class="record-section-title"><div><p class="section-label">PHOTO COLLECTION</p><h3>{{ tr('Photos of this person', '此人物的照片') }}</h3></div><span>{{ selectedPerson.photo_ids.length }} ITEMS</span></div><div class="detail-photos"><img v-for="id in selectedPerson.photo_ids" :key="id" :src="mediaUrl(photoById.get(id)?.url)" alt="Person photo" /></div></div><aside class="record-editor"><p class="section-label">PROFILE DETAILS</p><h3>{{ tr('Identity and notes', '身份与备注') }}</h3><label><span>{{ tr('Display name', '显示名称') }}</span><input v-model="editName" :placeholder="tr('Enter a name', '输入人物姓名')" /></label><label><span>{{ tr('Private note', '人物备注') }}</span><textarea v-model="editNote" :placeholder="tr('Add context about this person…', '添加关于此人的记录…')"></textarea></label><div class="llm-actions"><div><span>MEMORA AI</span><small>{{ health?.llm_configured ? tr('LLM connected', 'LLM 已配置') : tr('LLM setup required', '需要配置 LLM') }}</small></div><button type="button" :disabled="llmGenerating !== null" @click="generateDetail('name')"><span>✦</span>{{ llmGenerating === 'name' ? tr('Generating…', '生成中…') : tr('Generate name with LLM', '使用 LLM 生成名称') }}</button><button type="button" :disabled="llmGenerating !== null" @click="generateDetail('note')"><span>✦</span>{{ llmGenerating === 'note' ? tr('Generating…', '生成中…') : tr('Generate note with LLM', '使用 LLM 生成备注') }}</button></div><button class="primary-action" :disabled="busy" @click="saveDetail">{{ busy ? tr('Saving…', '保存中…') : tr('Save changes', '保存修改') }}</button></aside></div>
        </template>
        <template v-else>
          <div class="tool-heading"><p class="section-label">INSIGHTFACE CLUSTERS</p><h2>{{ tr('People who appear often', '照片里经常出现的人') }}</h2><p>{{ tr('Open a person to review their photos, set a name, and keep private notes.', '点击人物即可查看其全部照片、设置姓名并记录私人备注。') }}</p></div>
          <div v-if="!people.length" class="module-placeholder"><div class="large-symbol">●</div><div><h3>{{ tr('Run project-level people clustering', '运行项目级人物聚类') }}</h3><p>{{ tr('InsightFace analyzes faces locally. The first run may download a model; results stay within this project.', 'InsightFace 会在本机分析人脸。首次运行可能需要下载模型，结果只保存在当前项目。') }}</p></div><button class="primary-action" type="button" :disabled="busy || project.status !== 'ready'" @click="analyzePeople">{{ busy ? tr('Recognizing faces…', '正在识别人脸…') : tr('Analyze people', '开始人物分析') }}</button></div>
          <template v-else><div class="content-heading"><div><p class="section-label">PEOPLE INDEX</p><h2>{{ people.length }} {{ tr('people groups', '个人物分组') }}</h2></div><button class="primary-action" type="button" :disabled="busy" @click="analyzePeople">{{ tr('Re-run analysis', '重新分析人物') }}</button></div><div class="people-result-grid"><article v-for="person in people" :key="person.id" role="button" tabindex="0" @click="openPerson(person)" @keydown.enter="openPerson(person)"><img v-if="person.cover_url" :src="mediaUrl(person.cover_url)" alt="Person representative" /><div v-else class="person-fallback">●</div><div><strong>{{ person.name || tr(`Person ${person.id + 1}`, `人物 ${person.id + 1}`) }}</strong><span>{{ person.photo_ids.length }} {{ tr('photos · View profile →', '张照片 · 查看人物 →') }}</span></div></article></div></template>
        </template>
      </section>

      <section v-else-if="tab === 'events'" class="workspace-content narrow-content">
        <template v-if="selectedEvent || selectedJourney">
          <button class="detail-back" type="button" @click="selectedEvent = null; selectedJourney = null">← {{ tr('All events & journeys', '返回事件与旅程') }}</button>
          <header class="record-header story-header"><div class="story-mark">{{ selectedEvent ? 'E' : 'J' }}</div><div><p class="section-label">{{ selectedEvent ? 'EVENT DETAIL' : 'JOURNEY DETAIL' }}</p><h2>{{ (selectedEvent || selectedJourney)?.name || tr('Untitled story', '未命名故事') }}</h2><p>{{ dateLabel((selectedEvent || selectedJourney)?.start || null) }} — {{ dateLabel((selectedEvent || selectedJourney)?.end || null) }} · {{ (selectedEvent?.photo_ids || selectedJourney?.photo_ids || []).length }} {{ tr('photos', '张照片') }}</p></div></header>
          <div class="record-layout"><div class="record-gallery"><div class="record-section-title"><div><p class="section-label">STORY PHOTOS</p><h3>{{ selectedEvent ? tr('Event moments', '事件照片') : tr('Journey moments', '旅程照片') }}</h3></div><span>{{ (selectedEvent?.photo_ids || selectedJourney?.photo_ids || []).length }} ITEMS</span></div><div class="detail-photos"><img v-for="id in (selectedEvent?.photo_ids || selectedJourney?.photo_ids || [])" :key="id" :src="mediaUrl(photoById.get(id)?.url)" alt="Story photo" /></div></div><aside class="record-editor"><p class="section-label">STORY DETAILS</p><h3>{{ tr('Name and journal', '名称与记录') }}</h3><label><span>{{ tr('Story name', '故事名称') }}</span><input v-model="editName" :placeholder="tr('Name this moment', '为这段记忆命名')" /></label><label><span>{{ tr('Journal note', '旅程记录') }}</span><textarea v-model="editNote" :placeholder="tr('Write what you want to remember…', '记录你想保留的故事…')"></textarea></label><div class="llm-actions"><div><span>MEMORA AI</span><small>{{ health?.llm_configured ? tr('LLM connected', 'LLM 已配置') : tr('LLM setup required', '需要配置 LLM') }}</small></div><button type="button" :disabled="llmGenerating !== null" @click="generateDetail('name')"><span>✦</span>{{ llmGenerating === 'name' ? tr('Generating…', '生成中…') : tr('Generate name with LLM', '使用 LLM 生成名称') }}</button><button type="button" :disabled="llmGenerating !== null" @click="generateDetail('note')"><span>✦</span>{{ llmGenerating === 'note' ? tr('Generating…', '生成中…') : tr('Generate note with LLM', '使用 LLM 生成记录') }}</button></div><button class="primary-action" :disabled="busy" @click="saveDetail">{{ busy ? tr('Saving…', '保存中…') : tr('Save changes', '保存修改') }}</button></aside></div>
        </template>
        <template v-else>
          <div class="tool-heading"><p class="section-label">EVENT DISCOVERY</p><h2>{{ tr('Rediscover stories through time', '按时间重新看见故事') }}</h2><p>{{ tr('Events capture individual moments; journeys connect moments across time and place. Open any card to see its full story.', '事件记录单次重要时刻，旅程连接跨越时间与地点的多个片段。点击卡片即可查看完整内容。') }}</p></div>
          <div v-if="busy" class="loading-state">{{ tr('Discovering events…', '正在发现事件…') }}</div>
          <div v-else class="event-columns"><div><div class="column-heading"><div><p class="section-label">MOMENTS</p><h3>{{ tr('Events', '事件') }}</h3></div><span>{{ events.length }}</span></div><div class="event-stack"><article v-for="event in events" :key="event.id" role="button" tabindex="0" @click="openEvent(event)" @keydown.enter="openEvent(event)"><div class="event-date"><strong>{{ dateLabel(event.start) }}</strong><span>{{ event.photo_ids.length }} PHOTOS</span></div><div><h3>{{ event.name || tr(`Event ${event.id + 1}`, `事件 ${event.id + 1}`) }}</h3><p>{{ event.summary || `${dateLabel(event.start)} — ${dateLabel(event.end)}` }}</p></div><div class="event-thumbs"><img v-for="id in event.photo_ids.slice(0, 4)" :key="id" :src="mediaUrl(photoById.get(id)?.url)" alt="Event photo" /></div></article><div v-if="!events.length" class="module-placeholder">{{ tr('No events found yet.', '暂未发现事件。') }}</div></div></div><div><div class="column-heading"><div><p class="section-label">TRAVEL</p><h3>{{ tr('Journeys', '旅程') }}</h3></div><span>{{ journeys.length }}</span></div><div class="event-stack"><article v-for="journey in journeys" :key="journey.id" role="button" tabindex="0" @click="openJourney(journey)" @keydown.enter="openJourney(journey)"><div class="event-date"><strong>{{ dateLabel(journey.start) }}</strong><span>{{ journey.photo_ids.length }} PHOTOS</span></div><div><h3>{{ journey.name || tr(`Journey ${journey.id + 1}`, `旅程 ${journey.id + 1}`) }}</h3><p>{{ journey.destination_names?.join(' · ') || tr('Journey discovered from time and GPS', '根据时间和 GPS 发现的旅程') }}</p></div><div class="event-thumbs"><img v-for="id in journey.photo_ids.slice(0, 4)" :key="id" :src="mediaUrl(photoById.get(id)?.url)" alt="Journey photo" /></div></article><div v-if="!journeys.length" class="module-placeholder">{{ tr('Journeys need GPS metadata.', '旅程发现需要 GPS 元数据。') }}</div></div></div></div>
        </template>
      </section>

      <section v-else-if="tab === 'similar'" class="workspace-content narrow-content">
        <div class="tool-heading"><p class="section-label">SIMILAR SHOTS</p><h2>{{ tr('Clean up duplicates and bursts', '清理重复与连拍照片') }}</h2><p>{{ tr('Combine pHash, CLIP similarity, and capture time to find photos that belong to the same shot group.', '结合 pHash、CLIP 相似度和拍摄时间，找到属于同一组的相似照片。') }}</p></div>
        <div class="group-grid"><article v-for="group in groups" :key="group.id"><div class="group-preview"><img v-for="id in group.photo_ids.slice(0, 3)" :key="id" :src="mediaUrl(photoById.get(id)?.url)" alt="Similar photo" /></div><div><strong>{{ tr(`Similar group ${group.id + 1}`, `相似组 ${group.id + 1}`) }}</strong><span>{{ group.photo_ids.length }} {{ tr('photos · representative selected', '张照片 · 已推荐代表图') }}</span></div></article></div>
        <div v-if="!groups.length && !busy" class="module-placeholder">{{ tr('No similar-photo groups have been found in this project.', '当前项目尚未发现相似照片组。') }}</div>
      </section>

      <section v-else-if="tab === 'best'" class="workspace-content narrow-content">
        <div class="tool-heading"><p class="section-label">BEST SHOTS</p><h2>{{ tr('Keep the photos worth remembering', '挑出值得留下的照片') }}</h2><p>{{ tr('Rank the whole project using sharpness, exposure, face quality, and composition.', '综合清晰度、曝光、人物质量和构图，为整个项目建立精选列表。') }}</p></div>
        <div class="asset-grid best-grid"><article v-for="(photo, index) in bestShots" :key="photo.id"><div class="rank">{{ String(index + 1).padStart(2, '0') }}</div><img :src="mediaUrl(photo.url)" :alt="photo.filename" /><div><strong>{{ photo.filename }}</strong><span>{{ tr('Quality score', '质量评分') }} {{ Math.round(photo.score * 100) }}</span></div></article></div>
      </section>

      <section v-else class="workspace-content narrow-content">
        <div class="tool-heading"><p class="section-label">EXPORT CENTER</p><h2>{{ tr('Take organized results anywhere', '把整理结果带到下一步') }}</h2><p>{{ tr('Exports never modify your originals. Use them for backups, research, downstream tools, or another photo system.', '导出文件不会修改原始照片，适合备份、论文实验、二次开发或导入其他照片系统。') }}</p></div>
        <div class="export-grid">
          <a :href="exportUrl(project.id, 'manifest')"><span>JSON</span><div><h3>{{ tr('Complete project manifest', '完整项目清单') }}</h3><p>{{ tr('Photo metadata, quality scores, events, and similar-photo groups.', '照片元数据、质量评分、事件和相似照片分组。') }}</p><small>{{ tr('For backup and further processing', '适合备份和程序继续处理') }}</small></div><b>{{ tr('Download →', '下载 →') }}</b></a>
          <a :href="exportUrl(project.id, 'photos.csv')"><span>CSV</span><div><h3>{{ tr('Photo index table', '照片索引表') }}</h3><p>{{ tr('Filenames, capture times, GPS, and quality scores. Opens in Excel.', '文件名、拍摄时间、GPS 和质量评分，可用 Excel 打开。') }}</p><small>{{ tr('For review, labeling, and experiment statistics', '适合检查、标注与实验统计') }}</small></div><b>{{ tr('Download →', '下载 →') }}</b></a>
          <a :href="exportUrl(project.id, 'best-shots.zip')"><span>ZIP</span><div><h3>{{ tr('Best-shots collection', '最佳照片精选包') }}</h3><p>{{ tr('Top-ranked original photos with a complete score manifest.', '导出质量排名靠前的原图，并附带完整评分清单。') }}</p><small>{{ tr('For sharing, albums, and curation', '适合分享、制作相册与二次精选') }}</small></div><b>{{ tr('Download →', '下载 →') }}</b></a>
          <article><span>IMMICH</span><div><h3>{{ tr('Publish to an Immich album', '发布到 Immich 相册') }}</h3><p>{{ tr('Send selected results to Immich and continue with its timeline and album tools.', '把精选结果发送到 Immich，继续使用时间线和相册管理。') }}</p><small>{{ tr('For assets already synchronized with Immich', '适用于已经同步到 Immich 的资产') }}</small></div><b>{{ tr('Coming next', '后续接入') }}</b></article>
        </div>
      </section>
    </main>
  </div>
</template>
