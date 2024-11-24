import asyncio
import concurrent.futures
import json
import logging
import os
import time
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.ext import MessageHandler, filters

from llama_cpp import Llama


llm = Llama(
    # model_path="./qwen2.5-7b-instruct-q4_k_m.gguf",
    model_path="./qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    n_threads=8,
    n_gpu_layers=-1, # Uncomment to use GPU acceleration
    # n_gpu_layers=24,
    # seed=1337, # Uncomment to set a specific seed
    n_ctx=8192,
)
# llm = Llama.from_pretrained(
#     # repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
#     # filename="qwen2.5-3b-instruct-q4_k_m.gguf",
#     repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
#     filename="qwen2.5-7b-instruct-q4_k_m*.gguf",
#     # filename="qwen2.5-7b-instruct-q2_k.gguf",
#     verbose=False,
#     n_ctx=8192,
# )


def remove_unneeded_fields(message_dict):
    if not isinstance(message_dict, dict):
        return message_dict
    unneeded_fields = [
        key for key in message_dict.keys()
        if key.endswith('_id') or key.endswith('_date') or key.endswith('_chat_created') \
            or key in ("id", "date", "delete_chat_photo", "thumb", "thumbnail", "file_size", "type")
    ]
    return {
        k: remove_unneeded_fields(v)
        for k, v in message_dict.items() 
        if k not in unneeded_fields
    }

def ask_llm_is_it_spam(message: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    prompt = [
        {
            "role": "system",
            "content": (
                "Ты бот-модератор, который анализирует сообщения из публичного чата Obni(me) на предмет спама,"
                " рекламы и мошенничества. Obni(me) - это обнимательные встречи. В публичном чате люди обсуждают"
                " встречи, делятся друг с другом опытом и просто общаются. Реклама и объявления строго запрещены."
                " Если что-то, за что обычно платят деньги, предлагается бесплатно, то это тоже считается рекламой."
                "\n"
                "Ниже будет сообщение из публичного чата Obni(me) и его метаданные в формате JSON."
                " Ответь тоже в формате JSON на русском языке."
                "\n"
                "1. Сначала в поле \"contents\" порассуждай вслух о чём это сообщение. Опирайся на текст сообщения,"
                " если в метаданных упоминается файл, фотография или видео, то НЕ строй предположения о том,"
                " что на них изображено."
                "\n"
                "2. Затем в поле \"scenario\" придумай как автор сообщения планировал с помощью этого сообщения"
                " извлечь коммерческую выгоду из участников чата."
                "\n"
                "3. Затем в поле \"likelihood_and_severity\" порассуждай вслух похоже ли написанное в поле"
                " \"scenario\" на спам, рекламу, мошенничество или объявление с целью извлечения коммерческой выгоды."
                "\n"
                "4. Затем в поле \"is_commercial\" суммаризируй содержимое поля \"likelihood_and_severity\""
                " в одно слово: \"yes\", \"no\" или \"maybe\"."
            ),
        },
        {
            "role": "user", 
            "content": message,
        },
    ]
    logging.info(f"Prompt: {json.dumps(prompt, ensure_ascii=False, indent=2)}")
    prompt[-1]["content"] = json.dumps(prompt[-1]["content"], ensure_ascii=False)
    
    response = llm.create_chat_completion(
        temperature=0.0,
        messages=prompt,
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "contents": {"type": "string"},
                    "scenario": {"type": "string"},
                    "likelihood_and_severity": {"type": "string"},
                    "is_commercial": {"type": "string", "enum": ["yes", "no", "maybe"]},
                },
                "required": ["contents", "scenario", "likelihood_and_severity", "is_commercial"],
            },
        },
    )
    response = json.loads(response["choices"][0]["message"]["content"])
    logging.info(f"Response (took {time.time() - start_time:.2f}s): {json.dumps(response, ensure_ascii=False, indent=2)}")
    return response


async def on_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.info(f"Start command: {json.dumps(update.message.to_dict(), ensure_ascii=False, indent=2)}")
    await update.message.reply_text('Напиши мне сообщение, и я скажу тебе, спам это или нет.')

async def on_echo_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    llm_task = asyncio.get_event_loop().run_in_executor(
        None,
        ask_llm_is_it_spam,
        remove_unneeded_fields(update.message.to_dict())
    )

    while True:
        try:
            await update.message.chat.send_action("typing")
        except Exception as e:
            logging.warning(f"Error sending typing action: {e}")
        try:
            llm_response = await asyncio.wait_for(asyncio.shield(llm_task), timeout=4.0)
            break
        except asyncio.TimeoutError:
            continue
    
    llm_answer_to_russian = {"yes": "Точно", "no": "Не", "maybe": "Возможно"}
    await update.message.reply_text(
        (
            f"{llm_answer_to_russian[llm_response['is_commercial']]} спам"
            f"\n\n{llm_response['contents']}"
            f"\n\n{llm_response['scenario']}"
            f"\n\n{llm_response['likelihood_and_severity']}"
        ),
        reply_to_message_id=update.message.message_id
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    asyncio.get_event_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()
    application.add_handler(CommandHandler("start", on_start_command))
    application.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.VIDEO) & ~filters.COMMAND, on_echo_command))
    application.run_polling()

if __name__ == '__main__':
    main()
