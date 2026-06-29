/* BitsGPT — Frontend Logic */

const chatArea   = document.getElementById('chat-area');
const welcome    = document.getElementById('welcome');
const userInput  = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');
const errorToast = document.getElementById('error-toast');
const sugGrid    = document.getElementById('suggestions-grid');

let isStreaming = false;

// ── Suggestions ──────────────────────────────────────────────
async function loadSuggestions() {
  try {
    const res  = await fetch('/api/suggestions');
    const data = await res.json();
    sugGrid.innerHTML = data.questions
      .slice(0, 8)
      .map(q => `<button class="suggestion-btn" onclick="sendMessage(${JSON.stringify(q)})">${q}</button>`)
      .join('');
  } catch {
    // silently skip if server is offline
    sugGrid.innerHTML = [
      'What is the minimum CGPA?',
      'How does PS-1 and PS-2 work?',
      'What clubs can I join?',
      'What food outlets are at CP?',
      'What is ATMOS?',
      'How to apply for a minor?',
      "What does 'lite' mean?",
      'What are the fee components?',
    ]
      .map(q => `<button class="suggestion-btn" onclick="sendMessage(${JSON.stringify(q)})">${q}</button>`)
      .join('');
  }
}

loadSuggestions();

// ── Auto-resize textarea ──────────────────────────────────────
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
});

// ── Keyboard shortcuts ────────────────────────────────────────
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isStreaming) triggerSend();
  }
});

sendBtn.addEventListener('click', () => {
  if (!isStreaming) triggerSend();
});

function triggerSend() {
  const msg = userInput.value.trim();
  if (msg) sendMessage(msg);
}

// ── Send message ──────────────────────────────────────────────
async function sendMessage(text) {
  if (isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;

  // Hide welcome screen on first message
  if (welcome) welcome.style.display = 'none';

  // Append user bubble
  appendMessage('user', text);
  userInput.value = '';
  userInput.style.height = 'auto';

  // Append bot bubble with typing indicator
  const botBubble = appendMessage('bot', null);
  const contentEl = botBubble.querySelector('.bubble-content');
  contentEl.innerHTML = typingHTML();

  let fullText = '';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, stream: true }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'chunk') {
            fullText += event.text;
            contentEl.innerHTML = renderMarkdown(fullText) + cursorHTML();
          } else if (event.type === 'done') {
            contentEl.innerHTML = renderMarkdown(fullText);
            if (event.sources && event.sources.length > 0) {
              botBubble.querySelector('.bubble').appendChild(buildSources(event.sources));
            }
          } else if (event.type === 'error') {
            throw new Error(event.text);
          }
        } catch (parseErr) {
          // skip malformed SSE lines
        }
      }
    }

    // Remove cursor if still present
    contentEl.innerHTML = renderMarkdown(fullText);

  } catch (err) {
    contentEl.innerHTML = `<span style="color:#f87171">Sorry, something went wrong: ${escapeHtml(err.message)}</span>`;
    showError(err.message);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    userInput.focus();
    scrollToBottom();
  }
}

// ── DOM helpers ───────────────────────────────────────────────
function appendMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'bot' ? 'B' : 'U';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  if (text) content.innerHTML = renderMarkdown(text);

  bubble.appendChild(content);
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatArea.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function buildSources(sources) {
  const div = document.createElement('div');
  div.className = 'sources';
  div.innerHTML = '<div class="sources-label">Sources</div>';
  const seen = new Set();
  for (const s of sources.slice(0, 4)) {
    const label = s.chunk_title
      ? `${humanize(s.source)} — ${s.chunk_title}`
      : humanize(s.source);
    if (!seen.has(label)) {
      seen.add(label);
      const tag = document.createElement('span');
      tag.className = 'source-tag';
      tag.textContent = label;
      div.appendChild(tag);
    }
  }
  return div;
}

function humanize(src) {
  return src.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ── Light markdown renderer ───────────────────────────────────
function renderMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-size:0.9em">$1</code>')
    .replace(/^#{1,3} (.+)$/gm, '<strong>$1</strong>')
    .replace(/^[-*] (.+)$/gm, '• $1')
    .replace(/\n/g, '<br>');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function typingHTML() {
  return '<div class="typing"><span></span><span></span><span></span></div>';
}

function cursorHTML() {
  return '<span style="display:inline-block;width:2px;height:14px;background:var(--orange);margin-left:2px;vertical-align:middle;animation:pulse 0.7s infinite"></span>';
}

// ── Error toast ───────────────────────────────────────────────
let toastTimer;
function showError(msg) {
  errorToast.textContent = msg;
  errorToast.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { errorToast.style.display = 'none'; }, 5000);
}
