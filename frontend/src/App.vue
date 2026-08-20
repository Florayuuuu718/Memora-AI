<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from './api'
import ProjectHome from './components/ProjectHome.vue'
import ProjectWorkspace from './components/ProjectWorkspace.vue'
import type { HealthResponse, PhotoProject } from './types'

const projects = ref<PhotoProject[]>([])
const activeProject = ref<PhotoProject | null>(null)
const health = ref<HealthResponse | null>(null)

async function refreshProjects() {
  projects.value = (await api.projects()).projects
}

function openProject(project: PhotoProject) {
  activeProject.value = project
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

function updateProject(project: PhotoProject) {
  activeProject.value = project
  const index = projects.value.findIndex((value) => value.id === project.id)
  if (index >= 0) projects.value[index] = project
  else projects.value.unshift(project)
}

onMounted(async () => {
  const [healthResult] = await Promise.allSettled([api.health(), refreshProjects()])
  if (healthResult.status === 'fulfilled') health.value = healthResult.value
})
</script>

<template>
  <ProjectHome
    v-if="!activeProject"
    :projects="projects"
    :health="health"
    @open="openProject"
    @created="(project) => { updateProject(project); openProject(project) }"
  />
  <ProjectWorkspace
    v-else
    :project="activeProject"
    :health="health"
    @back="activeProject = null; refreshProjects()"
    @updated="updateProject"
  />
</template>
