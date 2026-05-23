from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod

from openai import OpenAI

from .config import OpenAIConfig
from .models import GenerationSlice, LlmTemplateRecord

ALLOWED_LABELS = [
    "person_name",
    "phone_kz",
    "email",
    "address_kz",
    "birth_date",
    "issue_date",
    "iin_kz",
    "bin_kz",
    "iban_kz",
    "card_pan",
    "card_last4",
    "contract_number",
    "loan_id",
    "sms_code",
    "otp",
    "cvv",
    "pin",
    "password",
    "code_word",
    "private_url",
]

MARKER_LABEL_MAPPING = {
    "{{NAME_1}}": "person_name",
    "{{PHONE_1}}": "phone_kz",
    "{{EMAIL_1}}": "email",
    "{{ADDRESS_1}}": "address_kz",
    "{{BIRTH_DATE_1}}": "birth_date",
    "{{ISSUE_DATE_1}}": "issue_date",
    "{{IIN_1}}": "iin_kz",
    "{{BIN_1}}": "bin_kz",
    "{{IBAN_1}}": "iban_kz",
    "{{CARD_PAN_1}}": "card_pan",
    "{{CARD_LAST4_1}}": "card_last4",
    "{{CONTRACT_1}}": "contract_number",
    "{{LOAN_ID_1}}": "loan_id",
    "{{SMS_1}}": "sms_code",
    "{{OTP_1}}": "otp",
    "{{CVV_1}}": "cvv",
    "{{PIN_1}}": "pin",
    "{{PASSWORD_1}}": "password",
    "{{CODE_WORD_1}}": "code_word",
    "{{URL_1}}": "private_url",
}

TEMPLATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["records"],
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["template", "values", "meta"],
                "properties": {
                    "template": {"type": "string"},
                    "values": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["label", "value"],
                            "properties": {
                                "label": {"type": "string", "enum": ALLOWED_LABELS},
                                "value": {"type": "string"},
                            },
                        },
                    },
                    "meta": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["language", "surface", "scenario", "pii_density", "noise_type", "hard_negative"],
                        "properties": {
                            "language": {"type": "string", "enum": ["ru", "kk", "ru_kk", "ru_kk_en"]},
                            "surface": {"type": "string", "enum": ["call_center_asr", "chat", "crm_note", "formal_notice", "hard_negative"]},
                            "scenario": {"type": "string"},
                            "pii_density": {"type": "string", "enum": ["none", "low", "medium", "high"]},
                            "noise_type": {"type": "string"},
                            "hard_negative": {"type": "boolean"},
                        },
                    },
                },
            },
        }
    },
}


class TemplateProvider(ABC):
    @abstractmethod
    def generate(self, generation_slice: GenerationSlice, *, seed: int) -> list[LlmTemplateRecord]:
        raise NotImplementedError


class OpenAITemplateProvider(TemplateProvider):
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        client_kwargs: dict[str, str] = {}
        api_key = config.resolved_api_key()
        base_url = config.resolved_base_url()
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, generation_slice: GenerationSlice, *, seed: int) -> list[LlmTemplateRecord]:
        prompt = build_prompt(generation_slice, seed=seed)
        response = self.client.responses.create(
            model=self.config.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "kz_pii_template_batch",
                    "strict": True,
                    "schema": TEMPLATE_SCHEMA,
                }
            },
        )
        payload = json.loads(response.output_text)
        return [LlmTemplateRecord.model_validate(item) for item in payload["records"]]


class FakeTemplateProvider(TemplateProvider):
    """Offline provider for tests and local smoke generation."""

    def generate(self, generation_slice: GenerationSlice, *, seed: int) -> list[LlmTemplateRecord]:
        rng = random.Random(seed)
        records: list[LlmTemplateRecord] = []
        for idx in range(generation_slice.count):
            values = _fake_values(seed * 100_000 + idx)
            if generation_slice.hard_negative or generation_slice.pii_density == "none":
                template = (
                    rng.choice(
                    [
                        "город Алматы работает в штатном режиме код ошибки 500 не относится к клиенту",
                        "отделение на Абая закрывается в 18:00 карта Visa Gold доступна клиентам",
                        "номер заявки 123456 принят оператором без персональных данных",
                    ]
                )
                    + f" синтетический пример {seed}-{idx}"
                )
                record_values = {}
            elif generation_slice.language == "kk":
                template = "сәлеметсіз бе менің жсн {{IIN_1}} атым {{NAME_1}} sms код {{SMS_1}}"
                record_values = {
                    "IIN_1": {"label": "iin_kz", "value": values["iin"]},
                    "NAME_1": {"label": "person_name", "value": values["name_kk"]},
                    "SMS_1": {"label": "sms_code", "value": values["sms"]},
                }
            elif generation_slice.surface == "chat":
                template = "здравствуйте поменяйте телефон {{PHONE_1}} и почту {{EMAIL_1}} по договору {{CONTRACT_1}}"
                record_values = {
                    "PHONE_1": {"label": "phone_kz", "value": values["phone"]},
                    "EMAIL_1": {"label": "email", "value": values["email"]},
                    "CONTRACT_1": {"label": "contract_number", "value": values["contract"]},
                }
            else:
                template = "алло здравствуйте меня зовут {{NAME_1}} мой иин {{IIN_1}} карта заканчивается на {{CARD_LAST4_1}}"
                record_values = {
                    "NAME_1": {"label": "person_name", "value": values["name_ru"]},
                    "IIN_1": {"label": "iin_kz", "value": values["iin"]},
                    "CARD_LAST4_1": {"label": "card_last4", "value": values["last4"]},
                }
            records.append(
                LlmTemplateRecord.model_validate(
                    {
                        "template": template,
                        "values": record_values,
                        "meta": generation_slice.model_dump(mode="json", exclude={"count"}),
                    }
                )
            )
        return records


SYSTEM_PROMPT = """Generate synthetic LLM-record batches in JSON for Kazakhstan banking/call-center privacy filtering.

You must return exactly one JSON object with one top-level key: "records".
"records" must be an array of record objects. Each record object must have top-level keys in this order:
"template", "values", "meta".

Each record must follow this structure:
{
  "template": "... {{MARKER_1}} ...",
  "values": {
    "MARKER_1": {"label": "label_from_list", "value": "synthetic_value"}
  },
  "meta": {
    "language": "ru|kk|ru_kk|ru_kk_en",
    "surface": "call_center_asr|chat|crm_note|formal_notice|hard_negative",
    "scenario": "banking_scenario",
    "pii_density": "none|low|medium|high",
    "noise_type": "clean|lowercase|no_punctuation|asr_noise|code_switch|fragmented",
    "hard_negative": true|false
  }
}

Do not generate spans, offsets, start/end fields, OPF labels, markdown, prose, or extra keys.

Template and marker rules:
- Use Kazakhstan banking, call-center, messenger support, formal notice, and CRM domains.
- For non-hard-negative records, include markers such as {{IIN_1}}, {{NAME_1}}, {{SMS_1}}.
- Every marker in template must have exactly one key in values.
- values must not contain markers that are absent from template.
- Do not reuse the same exact marker more than once in one template; for repeated entity types use {{NAME_1}}, {{NAME_2}}, etc.
- Do not put raw private values directly in template. Any private name, phone, email, address, ID, card, secret, URL, date, contract, or loan value must appear as a marker and as a values entry.
- Each marker prefix must match its values entry label according to the mapping supplied in the user input.
- Use only the allowed labels supplied in the user input.
- All values must be synthetic, plausible, fictional, and non-attributable to real people.
- Randomize marker combinations, wording, values, scenarios, and language style. Avoid repetitive templates and fixed placeholder-like values.

Hard negative rules:
- If meta.hard_negative=true, template must contain no private markers and values must be {}.
- Hard negatives may mention public entities and non-private lookalikes: "город Алматы", "отделение на Абая", error codes, public tariff names, Visa Gold, public bank websites.
- For hard negatives, meta.pii_density must be "none".

KZ banking context and style:
- Use Russian, Kazakh, mixed RU/KK, and RU/KK/EN banking-tech code switching as requested.
- Surfaces include ASR call transcripts, chat/messenger, CRM notes, formal notices, and hard negatives.
- Make examples realistic for Kazakhstan banking communication.
- Apply the requested noise_type: lowercase, no punctuation, ASR repeats, incomplete fragments, digit words in RU/KK, Cyrillic/Latin mixing, or clean text.
- Match the requested pii_density: none, low, medium, or high.

Good record examples inside the records array:
{
  "template": "здравствуйте это банк центр кредит клиент {{NAME_1}} иин {{IIN_1}} продиктуйте smscode {{SMS_1}} плиз",
  "values": {
    "NAME_1": {"label": "person_name", "value": "Айжан Сейдахметова"},
    "IIN_1": {"label": "iin_kz", "value": "030218300912"},
    "SMS_1": {"label": "sms_code", "value": "3948"}
  },
  "meta": {
    "language": "ru_kk_en",
    "surface": "call_center_asr",
    "scenario": "identity_verification",
    "pii_density": "medium",
    "noise_type": "asr_noise",
    "hard_negative": false
  }
}

{
  "template": "aidana емес клиент {{NAME_1}} телефон {{PHONE_1}} жсн {{IIN_1}} kartany bloktau card {{CARD_PAN_1}} pin code {{PIN_1}}",
  "values": {
    "NAME_1": {"label": "person_name", "value": "Айдана Жигерова"},
    "PHONE_1": {"label": "phone_kz", "value": "+77081231212"},
    "IIN_1": {"label": "iin_kz", "value": "021229551892"},
    "CARD_PAN_1": {"label": "card_pan", "value": "4405123478912345"},
    "PIN_1": {"label": "pin", "value": "2387"}
  },
  "meta": {
    "language": "ru_kk_en",
    "surface": "crm_note",
    "scenario": "card_blocking",
    "pii_density": "high",
    "noise_type": "code_switch",
    "hard_negative": false
  }
}
"""


def build_prompt(generation_slice: GenerationSlice, *, seed: int) -> str:
    return json.dumps(
        {
            "task": "generate_synthetic_kz_banking_pii_templates",
            "seed_hint": seed,
            "count": generation_slice.count,
            "language": generation_slice.language,
            "surface": generation_slice.surface,
            "scenario": generation_slice.scenario,
            "pii_density": generation_slice.pii_density,
            "noise_type": generation_slice.noise_type,
            "hard_negative": generation_slice.hard_negative,
            "allowed_markers": [
                *MARKER_LABEL_MAPPING,
            ],
            "marker_label_mapping": MARKER_LABEL_MAPPING,
            "allowed_labels": ALLOWED_LABELS,
            "output_contract": {
                "top_level": "Return exactly one JSON object: {'records': [record, ...]}.",
                "record_keys": ["template", "values", "meta"],
                "value_keys": ["label", "value"],
                "meta_keys": ["language", "surface", "scenario", "pii_density", "noise_type", "hard_negative"],
                "records_count_must_equal": generation_slice.count,
            },
        },
        ensure_ascii=False,
    )


def _fake_values(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    iin = f"{rng.randrange(0, 10**12):012d}"
    last4 = f"{rng.randrange(0, 10**4):04d}"
    return {
        "name_ru": rng.choice(["Айжан Сейдахметова", "Нурлан Абдрахманов", "Бауыржан Нурпеисов"]),
        "name_kk": rng.choice(["Айжан Сейдахметова", "Нұрлан Әбдірахманұлы", "Бауыржан Нұрпейісов"]),
        "iin": iin,
        "sms": f"{rng.randrange(0, 10**6):06d}",
        "last4": last4,
        "phone": f"+7 7{rng.randrange(0, 100):02d} {rng.randrange(0, 1000):03d} {rng.randrange(0, 100):02d} {rng.randrange(0, 100):02d}",
        "email": f"user{rng.randrange(1000, 9999)}@synthetic.kz",
        "contract": f"CN-{rng.randrange(100000, 999999)}",
    }


def provider_from_name(name: str, config: OpenAIConfig) -> TemplateProvider:
    if name == "fake":
        return FakeTemplateProvider()
    if name == "openai":
        return OpenAITemplateProvider(config)
    raise ValueError("provider must be 'openai' or 'fake'")
