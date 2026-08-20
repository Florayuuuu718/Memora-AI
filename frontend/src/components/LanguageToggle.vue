<script setup lang="ts">
import { locale, setLocale } from '../i18n'
</script>

<template>
  <div class="language-toggle" role="group" aria-label="Language">
    <button type="button" :class="{ active: locale === 'en' }" @click="setLocale('en')">EN</button>
    <button type="button" :class="{ active: locale === 'zh' }" @click="setLocale('zh')">中文</button>
  </div>
</template>
