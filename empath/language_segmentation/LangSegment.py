"""
This file bundles language identification functions.

See LICENSE file for more info.
https://github.com/adbar/py3langid

Projects:
https://github.com/juntaosun/LangSegment
"""

import os
import re
import sys
import numpy as np
from collections import Counter
from collections import defaultdict

# import langid
# import py3langid as langid
# pip install py3langid==0.2.2

# Enable language prediction probability normalization, the score of the probability prediction. Therefore, implement renormalization to produce output in the range of 0-1.
# langid disables probability normalization by default. For command-line usages of , it can be enabled by passing the flag.
# For probability normalization in library use, the user must instantiate their own . An example of such usage is as follows:
from py3langid.langid import LanguageIdentifier, MODEL_FILE

from empath.language_segmentation.utils.num import num2str

# -----------------------------------
# Changelog: The new version's word segmentation is more accurate.
# Changelog: The new version of the word segmentation is more accurate.
# Changelog: The new version's word segmentation is more accurate.
# Changelog: 분할이라는 단어의 새로운 버전이 더 정확합니다.
# -----------------------------------


# Word segmentation function:
# automatically identify and split the words (Chinese/English/Japanese/Korean) in the article or sentence according to different languages,
# making it more suitable for TTS processing.
# This code is designed for front-end text multi-lingual mixed annotation distinction, multi-language mixed training and inference of various TTS projects.
# This processing result is mainly for (Chinese = zh, Japanese = ja, English = en, Korean = ko), and can actually support up to 97 different language mixing processing.

# ===========================================================================================================
# Word segmentation function: Automatically recognizes and splits mixed languages (e.g. Chinese/English/Japanese/Korean) in texts to make them more suitable for TTS processing.
# This code is specially written for multi-language mixed annotation differentiation, mixed training, and inference in front-end text of various TTS projects.
# ===========================================================================================================
# (1) Automatic segmentation: "What do you read in Korean? Who is your PE teacher? In this presentation, four iPhone 15 series models were introduced."
# (2) Manual segmentation: "Is your name <ja>Sasaki?<ja>?"
# This processing result is mainly for (Chinese=zh, Japanese=ja, English=en, Korean=ko), but can actually support mixed processing of up to 97 different languages.
# ===========================================================================================================

# ===========================================================================================================
# 단어 분할 기능: 기사 또는 문장에서 단어(중국어/영어/일본어/한국어)를 다른 언어에 따라 자동으로 식별하고 분할하여 TTS 처리에 더 적합합니다.
# 이 코드는 프런트 엔드 텍스트 다국어 혼합 주석 분화, 다국어 혼합 교육 및 다양한 TTS 프로젝트의 추론을 위해 설계되었습니다.
# ===========================================================================================================
# (1) 자동 단어 분할: "한국어로 무엇을 읽습니까? 스포츠 씨? 이 컨퍼런스는 4개의 iPhone 15 시리즈 모델을 제공합니다."
# (2) 수동 참여: "이름이 <ja>Saki입니까? <ja>?"
# 이 처리 결과는 주로 (중국어 = zh, 일본어 = ja, 영어 = en, 한국어 = ko)를 위한 것이며 실제로 혼합 처리를 위해 최대 97개의 언어를 지원합니다.
# ===========================================================================================================

# ===========================================================================================================
# Word segmentation function: Automatically recognize and split articles or sentences according to different languages (such as Chinese/English/Japanese/Korean) to make them more suitable for TTS processing.
# This code is written specifically for multi-language mixed annotation differentiation, mixed training and inference of front-end text for various TTS projects.
# ===========================================================================================================
# (1) Automatic word segmentation: "How to read 오빠 in Korean? Who is your PE teacher? This launch event brought four iPhone 15 series models."
# (2) Manual word segmentation: "Is your name <ja>Sasaki?<ja>?"
# The processing results are mainly for (Chinese=zh, Japanese=ja, English=en, Korean=ko), and can actually support up to 97 different languages.
# ===========================================================================================================


# Manual segmentation tag specification: <language tag>text content</language tag>
# 수동 단어 분할 태그 사양: <언어 태그> 텍스트 내용</언어 태그>
# Manual word segmentation tag specification: <language tags> text content </language tags>
# Manual segmentation tag specification: <language tag>text content</language tag>
# ===========================================================================================================
# For manual word segmentation, labels need to appear in pairs, such as:
# For manual segmentation, tags need to appear in pairs, for example: "<ja>Sasaki<ja>" or "<ja>Sasaki</ja>"
# Error demonstration: "Your name is <ja>Sasaki." A single <ja> tag in this sentence will be ignored and will not be processed.
# Error demonstration: "Your name is <ja>佐々木。" Single <ja> tags that appear in this sentence will be ignored and will not be processed.
# ===========================================================================================================


# ===========================================================================================================
# Speech Synthesis Markup Language SSML, only its tags are supported here (not XML)
# Want to support more SSML tags? PRs are welcome!
# Note: Besides Chinese, it can be modified to support multilingual SSML, not just Chinese.
# Note: In addition to Chinese, it can also be modified to support multi-language SSML, not just Chinese.
# ===========================================================================================================
# Chinese implementation:
# [SSML] <number> = Chinese capitalized number reading (single character)
# [SSML] <telephone> = Convert numbers to uppercase Chinese characters for phone numbers (single character)
# [SSML] <currency> = Pronounce according to amount.
# [SSML] <date> = Pronounce according to date. Supports inputs like August 24, 2024, 2024/8/24, 2024-08, 08-24, 24, etc.
# ===========================================================================================================
class LangSSML:

    def __init__(self):
        # Pure numbers
        self._zh_numerals_number = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }

    # Standardize 2024/8/24, 2024-08, 08-24, 24 as "Year Month Day"
    # Standardize 2024/8/24, 2024-08, 08-24, 24 to "year-month-day"
    def _format_chinese_data(self, date_str: str):
        # Process date format
        input_date = date_str
        if date_str is None or date_str.strip() == "":
            return ""
        date_str = re.sub(r"[\/\._|年|月]", "-", date_str)
        date_str = re.sub(r"日", r"", date_str)
        date_arrs = date_str.split(" ")
        if len(date_arrs) == 1 and ":" in date_arrs[0]:
            time_str = date_arrs[0]
            date_arrs = []
        else:
            time_str = date_arrs[1] if len(date_arrs) >= 2 else ""

        def nonZero(num, cn, func=None):
            if func is not None:
                num = func(num)
            return f"{num}{cn}" if num is not None and num != "" and num != "0" else ""

        f_number = self.to_chinese_number
        f_currency = self.to_chinese_currency
        # year, month, day
        year_month_day = ""
        if len(date_arrs) > 0:
            year, month, day = "", "", ""
            parts = date_arrs[0].split("-")
            if len(parts) == 3:  # Format is YYYY-MM-DD
                year, month, day = parts
            elif len(parts) == 2:  # Format is MM-DD or YYYY-MM
                if len(parts[0]) == 4:  # Year-Month
                    year, month = parts
                else:
                    month, day = parts  # Month-Day
            elif len(parts[0]) > 0:  # Only Month-Day or Year
                if len(parts[0]) == 4:
                    year = parts[0]
                else:
                    day = parts[0]
            year, month, day = (
                nonZero(year, "年", f_number),
                nonZero(month, "月", f_currency),
                nonZero(day, "日", f_currency),
            )
            year_month_day = re.sub(r"([年|月|日])+", r"\1", f"{year}{month}{day}")
        # hours, minutes, seconds
        time_str = re.sub(r"[\/\.\-：_]", ":", time_str)
        time_arrs = time_str.split(":")
        hours, minutes, seconds = "", "", ""
        if len(time_arrs) == 3:  # H/M/S
            hours, minutes, seconds = time_arrs
        elif len(time_arrs) == 2:  # H/M
            hours, minutes = time_arrs
        elif len(time_arrs[0]) > 0:
            hours = f"{time_arrs[0]}点"  # H
        if len(time_arrs) > 1:
            hours, minutes, seconds = (
                nonZero(hours, "点", f_currency),
                nonZero(minutes, "分", f_currency),
                nonZero(seconds, "秒", f_currency),
            )
        hours_minutes_seconds = re.sub(
            r"([点|分|秒])+", r"\1", f"{hours}{minutes}{seconds}"
        )
        output_date = f"{year_month_day}{hours_minutes_seconds}"
        return output_date

    # [SSML] number = Chinese capitalized number reading (single character)
    # Chinese Numbers(single word)
    def to_chinese_number(self, num: str):
        pattern = r"(\d+)"
        zh_numerals = self._zh_numerals_number
        arrs = re.split(pattern, num)
        output = ""
        for item in arrs:
            if re.match(pattern, item):
                output += "".join(
                    zh_numerals[digit] if digit in zh_numerals else ""
                    for digit in str(item)
                )
            else:
                output += item
        output = output.replace(".", "点")
        return output

    # [SSML] telephone = Convert numbers to uppercase Chinese characters for phone numbers (single character)
    # Convert numbers to Chinese phone numbers in uppercase Chinese characters(single word)
    def to_chinese_telephone(self, num: str):
        output = self.to_chinese_number(num.replace("+86", ""))  # zh +86
        output = output.replace("一", "幺")
        return output

    # [SSML] currency = Pronounce according to amount.
    # Digital processing from GPT_SoVITS num.py （thanks）
    def to_chinese_currency(self, num: str):
        pattern = r"(\d+)"
        arrs = re.split(pattern, num)
        output = ""
        for item in arrs:
            if re.match(pattern, item):
                output += num2str(item)
            else:
                output += item
        output = output.replace(".", "点")
        return output

    # [SSML] date = Pronounce according to date. Supports inputs like August 24, 2024, 2024/8/24, 2024-08, 08-24, 24, etc.
    def to_chinese_date(self, num: str):
        chinese_date = self._format_chinese_data(num)
        return chinese_date


class LangSegment:

    def __init__(self):

        self.langid = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)

        self._text_cache = None
        self._text_lasts = None
        self._text_langs = None
        self._lang_count = None
        self._lang_eos = None

        # Customizable language matching tags:
        # Customizable language matching tags: These are supported，이 표현들은 모두 지지합니다
        # <zh>Hello<zh> , <ja>Sasaki</ja> , <en>OK<en> , <ko>Oppar</ko> these writings are all supported
        self.SYMBOLS_PATTERN = r"(<([a-zA-Z|-]*)>(.*?)<\/*[a-zA-Z|-]*>)"

        # Language filter group function, you can specify the languages to keep. Languages not in the filter group will be cleared. You can freely mix and match the languages supported by TTS speech synthesis.
        # 언어 필터 그룹 기능을 사용하면 예약된 언어를 지정할 수 있습니다. 필터 그룹에 없는 언어는 지워집니다. TTS 텍스트에서 지원하는 언어를 원하는 대로 일치시킬 수 있습니다.
        # Language filter group function, you can specify the languages to keep. Languages not in the filter group will be cleared. You can freely mix and match the languages supported by TTS speech synthesis.
        # The language filter group function allows you to specify reserved languages.
        # Languages not in the filter group will be cleared. You can match the languages supported by TTS Text To Speech as you like.
        # The higher the ranking, the higher the priority.

        # System default filter. (ISO 639-1 codes given)
        # ----------------------------------------------------------------------------------------------------------------------------------
        # "zh" Chinese, "en" English, "ja" Japanese, "ko" Korean, "fr" French, "vi" Vietnamese, "ru" Russian
        # "th" Thai
        # ----------------------------------------------------------------------------------------------------------------------------------
        self.DEFAULT_FILTERS = ["zh", "ja", "ko", "en"]

        # User-defined filters
        self.Langfilters = self.DEFAULT_FILTERS[:]  # Create copy

        # Merge text
        self.isLangMerge = True

        # Experimental: You can customize to add: "fr" French, "vi" Vietnamese.
        # Please use API to enable: self.setfilters(["zh", "en", "ja", "ko", "fr", "vi" , "ru" , "th"]) # Customize to add, e.g. "fr" French, "vi" Vietnamese.

        # Preview version function, automatically enabled or disabled, no need to set
        # Preview feature, automatically enabled or disabled, no settings required
        self.EnablePreview = False

        # In addition, it supports shorthand filters, just free combination by language.
        # In addition to that, it supports abbreviation filters, allowing for any combination of different languages.
        # Example: You can arbitrarily specify multiple combinations for filtering
        # Example: You can specify any combination to filter

        # Chinese/Japanese language priority threshold (score range 0 ~ 1): Enabled filters priority when score is lower than threshold <0.89.\n
        # 중/일본어 우선 순위 임계값(점수 범위 0-1): 점수가 설정된 임계값 <0.89보다 낮을 때 필터에서 우선 순위를 활성화합니다.
        # Chinese/Japanese language priority threshold (score range 0 ~ 1): Enabled filters priority when score is lower than threshold <0.89.\n
        # Chinese and Japanese language priority threshold (score range is 0 ~ 1): The default threshold is 0.89.  \n
        # Only the common characters between Chinese and Japanese are processed with confidence and priority. \n
        self.LangPriorityThreshold = 0.89

        # Langfilters = ["zh"]              # Recognize as Chinese
        # Langfilters = ["en"]              # Recognize as English
        # Langfilters = ["ja"]              # Recognize as Japanese
        # Langfilters = ["ko"]              # Recognize as Korean
        # Langfilters = ["zh_ja"]           # Chinese & Japanese mixed processing
        # Langfilters = ["zh_en"]           # Chinese & English mixed processing
        # Langfilters = ["ja_en"]           # Japanese & English mixed processing
        # Langfilters = ["zh_ko"]           # Chinese & Korean mixed processing
        # Langfilters = ["ja_ko"]           # Japanese & Korean mixed processing
        # Langfilters = ["en_ko"]           # English & Korean mixed processing
        # Langfilters = ["zh_ja_en"]        # Chinese, Japanese & English mixed processing
        # Langfilters = ["zh_ja_en_ko"]     # Chinese, Japanese, English & Korean mixed processing

        # For more filter combinations, please feel free to......
        # For more filter combinations, please feel free to......

        # Optional keep: Support Chinese numbers pinyin format, make frontend easy to implement phoneme tuning and inference, default is False.
        # Once enabled (True), the digital pinyin formats in parenthesis are preserved, and recognized as "zh" Chinese.
        self.keepPinyin = False

        # DEFINITION
        self.PARSE_TAG = re.compile(r"(⑥\$*\d+[\d]{6,}⑥)")

        self.LangSSML = LangSSML()

    def _clears(self):
        self._text_cache = None
        self._text_lasts = None
        self._text_langs = None
        self._text_waits = None
        self._lang_count = None
        self._lang_eos = None

    def _is_english_word(self, word):
        return bool(re.match(r"^[a-zA-Z]+$", word))

    def _is_chinese(self, word):
        for char in word:
            if "\u4e00" <= char <= "\u9fff":
                return True
        return False

    def _is_japanese_kana(self, word):
        pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]+")
        matches = pattern.findall(word)
        return len(matches) > 0

    def _insert_english_uppercase(self, word):
        modified_text = re.sub(r"(?<!\b)([A-Z])", r" \1", word)
        modified_text = modified_text.strip("-")
        return modified_text + " "

    def _split_camel_case(self, word):
        return re.sub(r"(?<!^)(?=[A-Z])", " ", word)

    def _statistics(self, language, text):
        # Language word statistics:
        # Chinese characters usually occupy double bytes
        if self._lang_count is None or not isinstance(self._lang_count, defaultdict):
            self._lang_count = defaultdict(int)
        lang_count = self._lang_count
        if not "|" in language:
            lang_count[language] += (
                int(len(text) * 2) if language == "zh" else len(text)
            )
        self._lang_count = lang_count

    def _clear_text_number(self, text):
        if text == "\n":
            return text, False  # Keep Line Breaks
        clear_text = re.sub(r"([^\w\s]+)", "", re.sub(r"\n+", "", text)).strip()
        is_number = len(re.sub(re.compile(r"(\d+)"), "", clear_text)) == 0
        return clear_text, is_number

    def _saveData(self, words, language: str, text: str, score: float, symbol=None):
        # Pre-detection
        clear_text, is_number = self._clear_text_number(text)
        # Merge the same language and save the results
        preData = words[-1] if len(words) > 0 else None
        if symbol is not None:
            pass
        elif preData is not None and preData["symbol"] is None:
            if len(clear_text) == 0:
                language = preData["lang"]
            elif is_number == True:
                language = preData["lang"]
            _, pre_is_number = self._clear_text_number(preData["text"])
            if preData["lang"] == language:
                self._statistics(preData["lang"], text)
                text = preData["text"] + text
                preData["text"] = text
                return preData
            elif pre_is_number == True:
                text = f'{preData["text"]}{text}'
                words.pop()
        elif is_number == True:
            priority_language = self._get_filters_string()[:2]
            if priority_language in "ja-zh-en-ko-fr-vi":
                language = priority_language
        data = {"lang": language, "text": text, "score": score, "symbol": symbol}
        filters = self.Langfilters
        if (
            filters is None
            or len(filters) == 0
            or "?" in language
            or language in filters
            or language in filters[0]
            or filters[0] == "*"
            or filters[0] in "alls-mixs-autos"
        ):
            words.append(data)
            self._statistics(data["lang"], data["text"])
        return data

    def _addwords(self, words, language, text, score, symbol=None):
        if text == "\n":
            pass  # Keep Line Breaks
        elif text is None or len(text.strip()) == 0:
            return True
        if language is None:
            language = ""
        language = language.lower()
        if language == "en":
            text = self._insert_english_uppercase(text)
        # text = re.sub(r'[(（）)]', ',' , text) # Keep it.
        text_waits = self._text_waits
        ispre_waits = len(text_waits) > 0
        preResult = text_waits.pop() if ispre_waits else None
        if preResult is None:
            preResult = words[-1] if len(words) > 0 else None
        if preResult and ("|" in preResult["lang"]):
            pre_lang = preResult["lang"]
            if language in pre_lang:
                preResult["lang"] = language = language.split("|")[0]
            else:
                preResult["lang"] = pre_lang.split("|")[0]
            if ispre_waits:
                preResult = self._saveData(
                    words,
                    preResult["lang"],
                    preResult["text"],
                    preResult["score"],
                    preResult["symbol"],
                )
        pre_lang = preResult["lang"] if preResult else None
        if ("|" in language) and (
            pre_lang and not pre_lang in language and not "…" in language
        ):
            language = language.split("|")[0]
        if "|" in language:
            self._text_waits.append(
                {"lang": language, "text": text, "score": score, "symbol": symbol}
            )
        else:
            self._saveData(words, language, text, score, symbol)
        return False

    def _get_prev_data(self, words):
        data = words[-1] if words and len(words) > 0 else None
        if data:
            return (data["lang"], data["text"])
        return (None, "")

    def _match_ending(self, input, index):
        if input is None or len(input) == 0:
            return False, None
        input = re.sub(r"\s+", "", input)
        if len(input) == 0 or abs(index) > len(input):
            return False, None
        ending_pattern = re.compile(r'([「」“”‘’"\':：。.！!?．？])')
        return ending_pattern.match(input[index]), input[index]

    def _cleans_text(self, cleans_text):
        cleans_text = re.sub(r"(.*?)([^\w]+)", r"\1 ", cleans_text)
        cleans_text = re.sub(r"(.)\1+", r"\1", cleans_text)
        return cleans_text.strip()

    def _mean_processing(self, text: str):
        if text is None or (text.strip()) == "":
            return None, 0.0
        arrs = self._split_camel_case(text).split(" ")
        langs = []
        for t in arrs:
            if len(t.strip()) <= 3:
                continue
            language, score = self.langid.classify(t)
            langs.append({"lang": language})
        if len(langs) == 0:
            return None, 0.0
        return Counter([item["lang"] for item in langs]).most_common(1)[0][0], 1.0

    def _lang_classify(self, cleans_text):
        language, score = self.langid.classify(cleans_text)
        # fix: Huggingface is np.float32
        if (
            score is not None
            and isinstance(score, np.generic)
            and hasattr(score, "item")
        ):
            score = score.item()
        score = round(score, 3)
        return language, score

    def _get_filters_string(self):
        filters = self.Langfilters
        return "-".join(filters).lower().strip() if filters is not None else ""

    def _parse_language(self, words, segment):
        LANG_JA = "ja"
        LANG_ZH = "zh"
        LANG_ZH_JA = f"{LANG_ZH}|{LANG_JA}"
        LANG_JA_ZH = f"{LANG_JA}|{LANG_ZH}"
        language = LANG_ZH
        regex_pattern = re.compile(r"([^\w\s]+)")
        lines = regex_pattern.split(segment)
        lines_max = len(lines)
        LANG_EOS = self._lang_eos
        for index, text in enumerate(lines):
            if len(text) == 0:
                continue
            EOS = index >= (lines_max - 1)
            nextId = index + 1
            nextText = lines[nextId] if not EOS else ""
            nextPunc = (
                len(re.sub(regex_pattern, "", re.sub(r"\n+", "", nextText)).strip())
                == 0
            )
            textPunc = (
                len(re.sub(regex_pattern, "", re.sub(r"\n+", "", text)).strip()) == 0
            )
            if not EOS and (
                textPunc == True or (len(nextText.strip()) >= 0 and nextPunc == True)
            ):
                lines[nextId] = f"{text}{nextText}"
                continue
            number_tags = re.compile(r"(⑥\d{6,}⑥)")
            cleans_text = re.sub(number_tags, "", text)
            cleans_text = re.sub(r"\d+", "", cleans_text)
            cleans_text = self._cleans_text(cleans_text)
            # fix:Langid's recognition of short sentences is inaccurate, and it is spliced longer.
            if not EOS and len(cleans_text) <= 2:
                lines[nextId] = f"{text}{nextText}"
                continue
            language, score = self._lang_classify(cleans_text)
            prev_language, prev_text = self._get_prev_data(words)
            if language != LANG_ZH and all(
                "\u4e00" <= c <= "\u9fff" for c in re.sub(r"\s", "", cleans_text)
            ):
                language, score = LANG_ZH, 1
            if len(cleans_text) <= 5 and self._is_chinese(cleans_text):
                filters_string = self._get_filters_string()
                if score < self.LangPriorityThreshold and len(filters_string) > 0:
                    index_ja, index_zh = filters_string.find(
                        LANG_JA
                    ), filters_string.find(LANG_ZH)
                    if index_ja != -1 and index_ja < index_zh:
                        language = LANG_JA
                    elif index_zh != -1 and index_zh < index_ja:
                        language = LANG_ZH
                if self._is_japanese_kana(cleans_text):
                    language = LANG_JA
                elif len(cleans_text) > 2 and score > 0.90:
                    pass
                elif EOS and LANG_EOS:
                    language = LANG_ZH if len(cleans_text) <= 1 else language
                else:
                    LANG_UNKNOWN = (
                        LANG_ZH_JA
                        if language == LANG_ZH
                        or (len(cleans_text) <= 2 and prev_language == LANG_ZH)
                        else LANG_JA_ZH
                    )
                    match_end, match_char = self._match_ending(text, -1)
                    referen = (
                        prev_language in LANG_UNKNOWN or LANG_UNKNOWN in prev_language
                        if prev_language
                        else False
                    )
                    if match_char in "。.":
                        language = (
                            prev_language if referen and len(words) > 0 else language
                        )
                    else:
                        language = f"{LANG_UNKNOWN}|…"
            text, *_ = re.subn(number_tags, self._restore_number, text)
            self._addwords(words, language, text, score)

    # ----------------------------------------------------------
    # [SSML] Chinese Number Processing (SSML support)
    # This processing is mainly for Chinese, used for dealing with SSML Chinese tags. Of course, it can support any language, for instance:
    # The default here is Chinese, which is used to process SSML Chinese tags. Of course, any language can be supported, for example:
    # Chinese phone numbers: <telephone>1234567</telephone>
    # Chinese digital numbers: <number>1234567</number>
    def _process_symbol_SSML(self, words, data):
        tag, match = data
        language = SSML = match[1]
        text = match[2]
        score = 1.0
        if SSML == "telephone":
            # Chinese- Phone number
            language = "zh"
            text = self.LangSSML.to_chinese_telephone(text)
        elif SSML == "number":
            # Chinese- Number reading
            language = "zh"
            text = self.LangSSML.to_chinese_number(text)
        elif SSML == "currency":
            # Chinese- Reading via amount
            language = "zh"
            text = self.LangSSML.to_chinese_currency(text)
        elif SSML == "date":
            # Chinese- Reading via amount
            language = "zh"
            text = self.LangSSML.to_chinese_date(text)
        self._addwords(words, language, text, score, SSML)

    # ----------------------------------------------------------
    def _restore_number(self, matche):
        value = matche.group(0)
        text_cache = self._text_cache
        if value in text_cache:
            process, data = text_cache[value]
            tag, match = data
            value = match
        return value

    def _pattern_symbols(self, item, text):
        if text is None:
            return text
        tag, pattern, process = item
        matches = pattern.findall(text)
        if len(matches) == 1 and "".join(matches[0]) == text:
            return text
        for i, match in enumerate(matches):
            key = f"⑥{tag}{i:06d}⑥"
            text = re.sub(pattern, key, text, count=1)
            self._text_cache[key] = (process, (tag, match))
        return text

    def _process_symbol(self, words, data):
        tag, match = data
        language = match[1]
        text = match[2]
        score = 1.0
        filters = self._get_filters_string()
        if language not in filters:
            self._process_symbol_SSML(words, data)
        else:
            self._addwords(words, language, text, score, True)

    def _process_english(self, words, data):
        tag, match = data
        text = match[0]
        filters = self._get_filters_string()
        priority_language = filters[:2]
        # Preview feature, other language segmentation processing
        enablePreview = self.EnablePreview
        if enablePreview == True:
            # Experimental: Other language support
            regex_pattern = re.compile(r"(.*?[。.?？!！]+[\n]{,1})")
            lines = regex_pattern.split(text)
            for index, text in enumerate(lines):
                if len(text.strip()) == 0:
                    continue
                cleans_text = self._cleans_text(text)
                language, score = self._lang_classify(cleans_text)
                if language not in filters:
                    language, score = self._mean_processing(cleans_text)
                if language is None or score <= 0.0:
                    continue
                elif language in filters:
                    pass  # pass
                elif score >= 0.95:
                    continue  # High score, but not in the filter, excluded.
                elif score <= 0.15 and filters[:2] == "fr":
                    language = priority_language
                else:
                    language = "en"
                self._addwords(words, language, text, score)
        else:
            # Default is English
            language, score = "en", 1.0
            self._addwords(words, language, text, score)

    def _process_Russian(self, words, data):
        tag, match = data
        text = match[0]
        language = "ru"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_Thai(self, words, data):
        tag, match = data
        text = match[0]
        language = "th"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_korean(self, words, data):
        tag, match = data
        text = match[0]
        language = "ko"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_quotes(self, words, data):
        tag, match = data
        text = "".join(match)
        childs = self.PARSE_TAG.findall(text)
        if len(childs) > 0:
            self._process_tags(words, text, False)
        else:
            cleans_text = self._cleans_text(match[1])
            if len(cleans_text) <= 5:
                self._parse_language(words, text)
            else:
                language, score = self._lang_classify(cleans_text)
                self._addwords(words, language, text, score)

    def _process_pinyin(self, words, data):
        tag, match = data
        text = match
        language = "zh"
        score = 1.0
        self._addwords(words, language, text, score)

    def _process_number(self, words, data):  # "$0" process only
        """
        Numbers alone cannot accurately identify language.
        Because numbers are universal in all languages.
        So it won't be executed here, just for testing.
        """
        tag, match = data
        language = words[0]["lang"] if len(words) > 0 else "zh"
        text = match
        score = 0.0
        self._addwords(words, language, text, score)

    def _process_tags(self, words, text, root_tag):
        text_cache = self._text_cache
        segments = re.split(self.PARSE_TAG, text)
        segments_len = len(segments) - 1
        for index, text in enumerate(segments):
            if root_tag:
                self._lang_eos = index >= segments_len
            if self.PARSE_TAG.match(text):
                process, data = text_cache[text]
                if process:
                    process(words, data)
            else:
                self._parse_language(words, text)
        return words

    def _merge_results(self, words):
        new_word = []
        for index, cur_data in enumerate(words):
            if "symbol" in cur_data:
                del cur_data["symbol"]
            if index == 0:
                new_word.append(cur_data)
            else:
                pre_data = new_word[-1]
                if cur_data["lang"] == pre_data["lang"]:
                    pre_data["text"] = f'{pre_data["text"]}{cur_data["text"]}'
                else:
                    new_word.append(cur_data)
        return new_word

    def _parse_symbols(self, text):
        TAG_NUM = "00"  # "00" => default channels , "$0" => testing channel
        TAG_S1, TAG_S2, TAG_P1, TAG_P2, TAG_EN, TAG_KO, TAG_RU, TAG_TH = (
            "$1",
            "$2",
            "$3",
            "$4",
            "$5",
            "$6",
            "$7",
            "$8",
        )
        TAG_BASE = re.compile(rf'(([【《（(“‘"\']*[LANGUAGE]+[\W\s]*)+)')
        # Get custom language filter
        filters = self.Langfilters
        filters = filters if filters is not None else ""
        # =======================================================================================================
        # Experimental: Other language support.Thử nghiệm: Hỗ trợ ngôn ngữ khác.Expérimental : prise en charge d’autres langues.
        # If characters for related language are missing, friends familiar with the language can submit patches out of missing phonetic symbols.
        # If relevant language characters are missing, friends who are familiar with the relevant languages can submit a submission to complete the missing pronunciation symbols.
        # S'il manque des caractères linguistiques pertinents, les amis qui connaissent les langues concernées peuvent soumettre une soumission pour compléter les symboles de prononciation manquants.
        # Nếu thiếu ký tự ngôn ngữ liên quan, những người bạn quen thuộc với ngôn ngữ liên quan có thể gửi bài để hoàn thành các ký hiệu phát âm còn thiếu.
        # -------------------------------------------------------------------------------------------------------
        # Preview feature, other language support
        enablePreview = self.EnablePreview
        if "fr" in filters or "vi" in filters:
            enablePreview = True
        self.EnablePreview = enablePreview
        # Experimental: French character support. Prise en charge des caractères français
        RE_FR = "" if not enablePreview else "àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ"
        # Experimental: Vietnamese character support. Hỗ trợ ký tự tiếng Việt
        RE_VI = (
            ""
            if not enablePreview
            else "đơưăáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựôâêơưỷỹ"
        )
        # -------------------------------------------------------------------------------------------------------
        # Basic options:
        process_list = [
            (
                TAG_S1,
                re.compile(self.SYMBOLS_PATTERN),
                self._process_symbol,
            ),  # Symbol Tag
            (
                TAG_KO,
                re.compile(re.sub(r"LANGUAGE", f"\uac00-\ud7a3", TAG_BASE.pattern)),
                self._process_korean,
            ),  # Korean words
            (
                TAG_TH,
                re.compile(re.sub(r"LANGUAGE", f"\u0e00-\u0e7f", TAG_BASE.pattern)),
                self._process_Thai,
            ),  # Thai words support.
            (
                TAG_RU,
                re.compile(re.sub(r"LANGUAGE", f"А-Яа-яЁё", TAG_BASE.pattern)),
                self._process_Russian,
            ),  # Russian words support.
            (
                TAG_NUM,
                re.compile(r"(\W*\d+\W+\d*\W*\d*)"),
                self._process_number,
            ),  # Number words, Universal in all languages, Ignore it.
            (
                TAG_EN,
                re.compile(
                    re.sub(r"LANGUAGE", f"a-zA-Z{RE_FR}{RE_VI}", TAG_BASE.pattern)
                ),
                self._process_english,
            ),  # English words + Other language support.
            (
                TAG_P1,
                re.compile(r'(["\'])(.*?)(\1)'),
                self._process_quotes,
            ),  # Regular quotes
            (
                TAG_P2,
                re.compile(
                    r"([\n]*[【《（(“‘])([^【《（(“‘’”)）》】]{3,})([’”)）》】][\W\s]*[\n]{,1})"
                ),
                self._process_quotes,
            ),  # Special quotes, There are left and right.
        ]
        # Extended options: Default False
        if self.keepPinyin == True:
            process_list.insert(
                1,
                (
                    TAG_S2,
                    re.compile(r"([\(（{](?:\s*\w*\d\w*\s*)+[}）\)])"),
                    self._process_pinyin,
                ),  # Chinese Pinyin Tag.
            )
        # -------------------------------------------------------------------------------------------------------
        words = []
        lines = re.findall(r".*\n*", re.sub(self.PARSE_TAG, "", text))
        for index, text in enumerate(lines):
            if len(text.strip()) == 0:
                continue
            self._lang_eos = False
            self._text_cache = {}
            for item in process_list:
                text = self._pattern_symbols(item, text)
            cur_word = self._process_tags([], text, True)
            if len(cur_word) == 0:
                continue
            cur_data = cur_word[0] if len(cur_word) > 0 else None
            pre_data = words[-1] if len(words) > 0 else None
            if (
                cur_data
                and pre_data
                and cur_data["lang"] == pre_data["lang"]
                and cur_data["symbol"] == False
                and pre_data["symbol"]
            ):
                cur_data["text"] = f'{pre_data["text"]}{cur_data["text"]}'
                words.pop()
            words += cur_word
        if self.isLangMerge == True:
            words = self._merge_results(words)
        lang_count = self._lang_count
        if lang_count and len(lang_count) > 0:
            lang_count = dict(
                sorted(lang_count.items(), key=lambda x: x[1], reverse=True)
            )
            lang_count = list(lang_count.items())
            self._lang_count = lang_count
        return words

    def setfilters(self, filters):
        # Clear the cache when the filter is changed
        # 필터가 변경되면 캐시를 지웁니다.
        # Clear the cache when the filter is changed
        # When the filter changes, clear the cache
        if self.Langfilters != filters:
            self._clears()
            self.Langfilters = filters

    def getfilters(self):
        return self.Langfilters

    def setPriorityThreshold(self, threshold: float):
        self.LangPriorityThreshold = threshold

    def getPriorityThreshold(self):
        return self.LangPriorityThreshold

    def getCounts(self):
        lang_count = self._lang_count
        if lang_count is not None:
            return lang_count
        text_langs = self._text_langs
        if text_langs is None or len(text_langs) == 0:
            return [("zh", 0)]
        lang_counts = defaultdict(int)
        for d in text_langs:
            lang_counts[d["lang"]] += (
                int(len(d["text"]) * 2) if d["lang"] == "zh" else len(d["text"])
            )
        lang_counts = dict(
            sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
        )
        lang_counts = list(lang_counts.items())
        self._lang_count = lang_counts
        return lang_counts

    def getTexts(self, text: str):
        if text is None or len(text.strip()) == 0:
            self._clears()
            return []
        # lasts
        text_langs = self._text_langs
        if self._text_lasts == text and text_langs is not None:
            return text_langs
        # parse
        self._text_waits = []
        self._lang_count = None
        self._text_lasts = text
        text = self._parse_symbols(text)
        self._text_langs = text
        return text

    def classify(self, text: str):
        return self.getTexts(text)


def printList(langlist):
    """
    Function: Print the array results
    기능: 어레이 결과 인쇄
    Function: Print the array results
    Function: Print array results
    """
    print("\n===================[Print Result]===================")
    if langlist is None or len(langlist) == 0:
        print("No content result")
        return
    for line in langlist:
        print(line)
    pass


def main():

    # -----------------------------------
    # Changelog: The new version's word segmentation is more accurate.
    # Changelog: The new version of the word segmentation is more accurate.
    # Changelog: The new version's word segmentation is more accurate.
    # Changelog: 분할이라는 단어의 새로운 버전이 더 정확합니다.
    # -----------------------------------

    # Input Example 1: (including Japanese, Chinese)
    # text = "“昨日は雨が降った，音楽、映画。。。”你今天学习日语了吗？春は桜の季節です。语种分词是语音合成必不可少的环节。言語分詞は音声合成に欠かせない環節である！"

    # Input Example 2: (including Japanese, Chinese)
    # text = "欢迎来玩。東京，は日本の首都です。欢迎来玩.  太好了!"

    # Input Example 3: (including Japanese, Chinese)
    # text = "明日、私たちは海辺にバカンスに行きます。你会说日语吗：“中国語、話せますか” 你的日语真好啊！"

    # Input Example 4: (including Japanese, Chinese, Korean, English)
    # text = "你的名字叫<ja>佐々木？<ja>吗？韩语中的안녕 오빠读什么呢？あなたの体育の先生は誰ですか? 此次发布会带来了四款iPhone 15系列机型和三款Apple Watch等一系列新品，这次的iPad Air采用了LCD屏幕"

    # Experimental: "fr" French, "vi" Vietnamese, "ru" Russian, "th" Thai support.
    langsegment = LangSegment()
    langsegment.setfilters(["fr", "vi", "ja", "zh", "ko", "en", "ru", "th"])
    text = """
I like listening to music on rainy days.
I enjoy listening to music on rainy days.
I like listening to music on rainy days.
비 오는 날에 음악을 듣는 것을 즐깁니다。
J'aime écouter de la musique les jours de pluie.
Tôi thích nghe nhạc vào những ngày mưa.
Мне нравится слушать музыку в дождливую погоду.
ฉันชอบฟังเพลงในวันที่ฝนตก
"""

    # Segmentation: (Only one line of code is required to access the TTS project)
    langlist = langsegment.getTexts(text)
    printList(langlist)

    # Language statistics:
    print("\n===================[Language statistics]===================")
    # Get all language array results, sort by content word count in descending order
    # Get the array results in all languages, sorted in descending order according to the number of content words
    langCounts = langsegment.getCounts()
    print(langCounts, "\n")

    # Get the main language of the content according to the result (Language, word count including punctuation)
    # Get the main language of content based on the results (language, word count including punctuation)
    lang, count = langCounts[0]
    print(f"The main language of the input content is = {lang}, Word count = {count}")
    print("==================================================\n")

    # Word output: lang = language, text = content
    # ===================[Print Result]===================
    # {'lang': 'zh', 'text': '你的名字叫'}
    # {'lang': 'ja', 'text': '佐々木？'}
    # {'lang': 'zh', 'text': '吗？韩语中的'}
    # {'lang': 'ko', 'text': '안녕 오빠'}
    # {'lang': 'zh', 'text': '读什么呢？'}
    # {'lang': 'ja', 'text': 'あなたの体育の先生は誰ですか?'}
    # {'lang': 'zh', 'text': ' 此次发布会带来了四款'}
    # {'lang': 'en', 'text': 'i Phone  '}
    # {'lang': 'zh', 'text': '15系列机型和三款'}
    # {'lang': 'en', 'text': 'Apple Watch '}
    # {'lang': 'zh', 'text': '等一系列新品，这次的'}
    # {'lang': 'en', 'text': 'i Pad Air '}
    # {'lang': 'zh', 'text': '采用了'}
    # {'lang': 'en', 'text': 'L C D '}
    # {'lang': 'zh', 'text': '屏幕'}
    # ===================[Language statistics]===================

    # ===================[Language statistics]===================
    # [('zh', 51), ('ja', 19), ('en', 18), ('ko', 5)]

    # 输入内容的主要语言为 = zh ，字数 = 51
    # ==================================================
    # The main language of the input content is = zh, word count = 51


if __name__ == "__main__":
    main()
