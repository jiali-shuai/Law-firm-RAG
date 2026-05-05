const API_BASE = '/api'

export async function fetchCollections() {
  const res = await fetch(`${API_BASE}/collections`)
  if (!res.ok) throw new Error('请求失败')
  return res.json()
}

export async function createCollection(name, description) {
  const res = await fetch(`${API_BASE}/collections`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description }),
  })
  if (!res.ok) {
    const data = await res.json()
    throw new Error(data.detail || '创建失败')
  }
  return res.json()
}

export async function fetchDocuments() {
  const res = await fetch(`${API_BASE}/documents`)
  if (!res.ok) throw new Error('请求失败')
  return res.json()
}

export async function uploadDocument(file, collectionName) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('collection_name', collectionName)
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const data = await res.json()
    throw new Error(data.detail || '上传失败')
  }
  return res.json()
}

export async function deleteDocument(fileName) {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(fileName)}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    const data = await res.json()
    throw new Error(data.detail || '删除失败')
  }
  return res.json()
}

export async function sendLegalChat(question, sessionId, paramOverrides = {}) {
  const res = await fetch(`${API_BASE}/legal/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      param_overrides: paramOverrides,
    }),
  })
  if (!res.ok) {
    const data = await res.json()
    throw new Error(data.detail || '法律咨询失败')
  }
  return res.json()
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}
