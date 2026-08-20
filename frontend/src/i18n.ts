import { ref } from 'vue'

export type Locale = 'en' | 'zh'

const saved = window.localStorage.getItem('memora-language')
export const locale = ref<Locale>(saved === 'zh' ? 'zh' : 'en')

export function setLocale(value: Locale) {
  locale.value = value
  window.localStorage.setItem('memora-language', value)
  document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en'
}

export function tr(english: string, chinese: string): string {
  return locale.value === 'zh' ? chinese : english
}

setLocale(locale.value)
