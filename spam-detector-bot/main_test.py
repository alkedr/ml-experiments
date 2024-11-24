from typing import Any, Dict, List
import json
from main import ask_llm_is_it_spam


def assert_ask_llm_is_it_spam_returns(expected_is_commercial: List[str], message: Dict[str, Any]):
    llm_response = ask_llm_is_it_spam(message)
    assert llm_response['is_commercial'] in expected_is_commercial, \
        f"Expected {expected_is_commercial} but got {llm_response['is_commercial']}," \
        f" full response: {json.dumps(llm_response, indent=2, ensure_ascii=False)}"


def test_spam_200k_per_month():
    assert_ask_llm_is_it_spam_returns(['yes'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        },
      },
      "text": "Зарабатывай до 200000 р. в месяц не вставая с дивана! Проверенный способ! Пиши в личку!",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr",
        "language_code": "en",
        "is_premium": True
      },
    })


def test_obnime_announcement():
    assert_ask_llm_is_it_spam_returns(['yes', 'maybe'], {
      "entities": [
        {
          "length": 9,
          "offset": 26,
          "type": "hashtag"
        }
      ],
      "forward_origin": {
        "sender_user": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        }
      },
      "text": "Просто напоминалочка, что #ЭКСТАТИК - уже на следующей неделе 💜\n\nCамое время ставить себе в календарики и бронить билетик: \n16 ноября (сб), 17:00-21:00, стоимость 2600₽\n\nДля тех, кто не знает, что такое Ecstatic Dance и почему у нас - самые крутые экстатики, рассказываем:\n\n💜Что такое Экстатик дэнс? \nЭто телесная практика свободного танца, форма медитации в движении, которая, через музыку и атмосферу позволяет соединиться с собственным телом, распаковать зажатую энергию, прожить накопленные эмоции, гармонизировать ум и тело\n\n💜Чем наша практика отличается от других похожих?\nУ нас есть Митяй! Который не только народно-любимый душенька и заботушка, но и очень чуткий и невероятно талантливый музыкант (кто был на КМСах и выездах подтвердят)💜. Своей музыкой Митя очень деликатно и бережно перебирает самые глубокие и нежные струны души и возвращает к самому себе + будет разминка и бережное сопровождение вглубь и обратно.\n\n💜Как подготовиться?\nВо-первых, записаться. Мы не записываем много народу, чтобы можно было комфортно двигаться. Во-вторых, взять с собой удобную для движения сменную одежду и носочки. В пространстве будет вода, раздевалки и душевые, но одежду точно лучше взять с собой - после практики вас можно будет выжимать💜 Ну и, плотно не кушать за пару часов до💜Это всё. \n\n💜Нужно ли уметь танцевать и есть ли ограничения?\nНет, не нужно. Практика свободного танца тем и хороша, что в ней нет правильного/не правильного, нет привычных движений/жанров/привычной музыки. Есть только ваше тело, звуки, которые приглашают в путешествие и тело само выберет, как ему хочется двигаться и как реагировать. Из ограничений, пожалуй, только высокая склонность к эпилепсии или проблемы со слухом, потому что музыка будет громкой. \n\nВсё, я готов записаться / я не увидел ответа на свой вопрос: пиши трогательным менеджерам",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr",
        "language_code": "en",
        "is_premium": True
      }
    })


def test_buy_elephant():
    assert_ask_llm_is_it_spam_returns(['no'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        }
      },
      "text": "Купи слона",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr",
        "language_code": "en",
        "is_premium": True
      }
    })


def test_random1():
    assert_ask_llm_is_it_spam_returns(['no'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Alexandra",
          "is_bot": False,
          "is_premium": True,
          "username": "Aleksandra_Iskakova"
        }
      },
      "text": "Экстааатиииик🤩🤩🤩 еееее, дождалась 🥳",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Alexandra",
        "username": "Aleksandra_Iskakova",
        "is_premium": True
      }
    })


def test_extatologist():
    assert_ask_llm_is_it_spam_returns(['no', 'maybe'], {
      "caption": "Опытный экстатолог обучает новичка",
      "forward_origin": {
        "sender_user": {
          "first_name": "Ольга Gorbunova",
          "is_bot": False,
          "is_premium": True,
          "username": "olgagorbunova11"
        }
      },
      "video": {
        "duration": 5,
        "height": 976,
        "mime_type": "video/mp4",
        "width": 1280
      },
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Ольга Gorbunova",
        "username": "olgagorbunova11",
        "is_premium": True
      }
    })


def test_reply_to_extatologist():
    assert_ask_llm_is_it_spam_returns(['no'], {
      "forward_origin": {
        "sender_user_name": "Alex Rimcy (SPb)"
      },
      "reply_to_message": {
        "caption": "Опытный экстатолог обучает новичка",
        "forward_origin": {
          "sender_user": {
            "first_name": "Ольга Gorbunova",
            "is_bot": False,
            "is_premium": True,
            "username": "olgagorbunova11"
          }
        },
        "video": {
          "duration": 5,
          "height": 976,
          "mime_type": "video/mp4",
          "width": 1280
        },
        "chat": {
          "first_name": "Александр",
          "last_name": "Кедрик",
          "username": "alkedr"
        },
        "from": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        },
        "forward_from": {
          "is_bot": False,
          "first_name": "Ольга Gorbunova",
          "username": "olgagorbunova11",
          "is_premium": True
        }
      },
      "text": "— Сколько раз посмотрел?\n— Да",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_sender_name": "Alex Rimcy (SPb)"
    })


def test_random2():
    assert_ask_llm_is_it_spam_returns(['no'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Голосовой",
          "is_bot": False,
          "last_name": "Вредитель",
          "username": "irvins_cass"
        }
      },
      "text": "Было так тепло на встрече, что аж пожарная тревога включилась",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Голосовой",
        "last_name": "Вредитель",
        "username": "irvins_cass"
      }
    })


def test_spam_real_money():
    assert_ask_llm_is_it_spam_returns(['yes'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        }
      },
      "text": "Быстро заработать реальные деньги! Пиши в личку!",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr",
        "language_code": "en",
        "is_premium": True
      }
    })


def test_spam_get_rich_fast():
    assert_ask_llm_is_it_spam_returns(['yes'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Александр",
          "is_bot": False,
          "is_premium": True,
          "language_code": "en",
          "last_name": "Кедрик",
          "username": "alkedr"
        }
      },
      "text": "Get rich fast! Call +7962432523 now!",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr",
        "language_code": "en",
        "is_premium": True
      }
    })


def test_spam_hiring():
    assert_ask_llm_is_it_spam_returns(['yes'], {
      "forward_origin": {
        "sender_user": {
          "first_name": "Bellial",
          "is_bot": False,
          "username": "XBellialoffX"
        }
      },
      "text": "🔥ΗАБ0Ρ С0ТΡУДΗИКΟΒ🔥\n\n  Что мы ΒΑΜ предлагаем 🫵\n⭕️ Κарьерный ρост ↗️\n⭕️ Βысокая заρплата от 250.000р в месяц 💰\n⭕️ Гибκий графиκ рабοты 🗓\n⭕️ Οплачиваемая стажиροвка💸\n⭕️ Выплаты 2 ρазы в неделю 🔝\n⭕️ Οпыт не обязателен (всему научим )‼️\n\nΠИШИ Β ЛИЧКУ И УЗНΑЙ БΟЛЬШЕ",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_from": {
        "is_bot": False,
        "first_name": "Bellial",
        "username": "XBellialoffX"
      }
    })


def test_spam_meditations_for_women():
    assert_ask_llm_is_it_spam_returns(['yes'], {
      "entities": [
        {
          "length": 7,
          "offset": 0,
          "type": "bold"
        },
        {
          "length": 7,
          "offset": 0,
          "type": "underline"
        },
        {
          "length": 42,
          "offset": 12,
          "type": "bold"
        },
        {
          "length": 42,
          "offset": 12,
          "type": "italic"
        },
        {
          "length": 21,
          "offset": 344,
          "type": "bold"
        },
        {
          "length": 7,
          "offset": 368,
          "type": "text_link",
          "url": "https://t.me/kseniyakropocheva_bot?start=dl-1726861650fe6c0d9180ba"
        },
        {
          "length": 41,
          "offset": 379,
          "type": "underline"
        },
        {
          "length": 8,
          "offset": 444,
          "type": "text_link",
          "url": "https://t.me/kseniyakropocheva_bot?start=dl-17268570346df3f0b57285"
        },
        {
          "length": 8,
          "offset": 480,
          "type": "text_link",
          "url": "https://t.me/kseniyakropocheva_bot?start=dl-1726858318092110b69572"
        },
        {
          "length": 8,
          "offset": 514,
          "type": "text_link",
          "url": "https://t.me/kseniyakropocheva_bot?start=dl-17268595857f2adfbb794d"
        },
        {
          "length": 56,
          "offset": 527,
          "type": "bold"
        },
        {
          "length": 56,
          "offset": 527,
          "type": "italic"
        }
      ],
      "forward_origin": {
        "sender_user_name": "Deleted Account"
      },
      "link_preview_options": {
        "is_disabled": True
      },
      "text": "Подарок 🎁\n\nМедитации для женщин, которые помогут Вам:\n\n💟 Наладить доверительные отношения с детьми;\n💟Вернуть гармонию и спокойствие в родительские отношения;\n💟Беременным женщинам помогут выносить и родить здорового малыша;\n💟Женщинам, которые не могут забеременеть, настроят на успешное зачатие. \n\n📌По ссылке переходите в бота и забирайте БЕСПЛАТНЫЕ МЕДИТАЦИИ ➡️ перейти \n\n\nТакже сейчас бесплатно доступны медитации:\n\n✅ Исцеляющий импульс получить🎁 \n✅ Здоровье и долголетие получить 🎁\n✅Привлечение клиентов получить 🎁\n\nСлушайте медитации и прочувствуйте на себе их мощность ⚡",
      "chat": {
        "first_name": "Александр",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "from": {
        "first_name": "Александр",
        "is_bot": False,
        "is_premium": True,
        "language_code": "en",
        "last_name": "Кедрик",
        "username": "alkedr"
      },
      "forward_sender_name": "Deleted Account"
    })
