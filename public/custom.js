/* 修改输入框提示文本为中文 */
setInterval(function () {
  var input = document.querySelector('#chat-input textarea, #chat-input [contenteditable], [data-placeholder]');
  if (input) {
    if (input.getAttribute('placeholder') !== '输入法律问题，或上传文件') {
      input.setAttribute('placeholder', '输入法律问题，或上传文件');
    }
    if (input.getAttribute('data-placeholder') !== '输入法律问题，或上传文件') {
      input.setAttribute('data-placeholder', '输入法律问题，或上传文件');
    }
  }
  // 也尝试找 Chainlit 的 composer
  var composer = document.querySelector('[class*="composer"] textarea, [class*="Composer"] textarea');
  if (composer && composer.placeholder !== '输入法律问题，或上传文件') {
    composer.placeholder = '输入法律问题，或上传文件';
  }
}, 800);
