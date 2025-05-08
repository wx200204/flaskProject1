import pyttsx3
import queue
import threading
from threading import Thread

class VoiceFeedback:
    """异步语音提示模块"""
    
    def __init__(self, rate=180, volume=0.9):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)  # 语速
        self.engine.setProperty('volume', volume)  # 音量
        
        # 尝试设置中文语音
        try:
            voices = self.engine.getProperty('voices')
            chinese_voice = None
            
            # 防止空列表或索引错误
            if voices:
                for voice in voices:
                    try:
                        if hasattr(voice, 'languages') and voice.languages and ('chinese' in voice.languages[0].lower() or 'zh' in voice.id.lower()):
                            chinese_voice = voice.id
                            break
                    except (IndexError, AttributeError) as e:
                        print(f"处理语音时出错: {str(e)}")
                    
            if chinese_voice:
                self.engine.setProperty('voice', chinese_voice)
        except Exception as e:
            print(f"设置语音引擎失败: {str(e)}")
            
        # 语音消息队列
        self.message_queue = queue.Queue()
        self.running = False
        self.last_messages = set()  # 用于避免重复语音提示
        
    def start(self):
        """启动语音提示线程"""
        if self.running:
            return
            
        self.running = True
        self.voice_thread = Thread(target=self._process_voice_queue, daemon=True)
        self.voice_thread.start()
        print("语音提示模块已启动")
        
    def _process_voice_queue(self):
        """处理语音队列的线程函数"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.5)
                if message:
                    # 避免重复提示相同内容
                    if message not in self.last_messages:
                        self.engine.say(message)
                        self.engine.runAndWait()
                        
                        # 更新最近消息记录
                        self.last_messages.add(message)
                        if len(self.last_messages) > 5:  # 只保留最近5条消息
                            self.last_messages.pop()
                            
                self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"语音提示错误: {str(e)}")
                
    def speak(self, message, priority=False):
        """添加语音提示到队列"""
        if not message:
            return
            
        # 对于高优先级消息，清空当前队列
        if priority:
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
                    
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            pass
            
    def stop(self):
        """停止语音提示线程"""
        self.running = False
        if self.voice_thread and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=1.0)
            
        # 清空队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except queue.Empty:
                break 