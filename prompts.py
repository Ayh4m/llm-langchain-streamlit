from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    # SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    # AIMessagePromptTemplate
)
from langchain.schema import SystemMessage
from langchain.output_parsers import CommaSeparatedListOutputParser


# Output Parsers
output_parser = CommaSeparatedListOutputParser()

# System Message
system_msg_prompt = SystemMessage(content="You are a helpful assistant that can answer questions about an industry")
# or
# system_template = "You are a helpful assistant that can answer questions about an industry"
# system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human Prompts
human_prompts_and_parsers_and_headings = [
    (
        PromptTemplate(
            template="What is the definition of the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Definition"
    ),
    (
        PromptTemplate(
            template="What is the primary focus or purpose of the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Main focus/purpose"
    ),
    (
        PromptTemplate(
            template="Where are the majority of businesses in the {industry_title} industry located in the US?"
                     "\n{format_instructions}",
            input_variables=["industry_title"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        ),
        output_parser,
        "Locations of main businesses in the US"
    ),
    (
        PromptTemplate(
            template="Who is the target customer base for the {industry_title} industry?\n{format_instructions}",
            input_variables=["industry_title"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        ),
        output_parser,
        "Target Customers"
    ),
    (
        PromptTemplate(
            template="What skills or qualifications are typically needed for careers in the {industry_title} industry?"
                     "\n{format_instructions}",
            input_variables=["industry_title"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        ),
        output_parser,
        "Required skills and qualifications"
    ),
    (
        PromptTemplate(
            template="How has the {industry_title} industry evolved over time?",
            input_variables=["industry_title"]
        ),
        None,
        "Development over time"
    ),
    (
        PromptTemplate(
            template="How does the supply chain work in the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Supply chain"
    ),
    (
        PromptTemplate(
            template="What are the key financial indicators of the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Key financial indicators"
    ),
    (
        PromptTemplate(
            template="What are the current trends shaping the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Current shaping trends"
    ),
    (
        PromptTemplate(
            template="How is the {industry_title} industry affected by economic factors "
                     "such as inflation, unemployment rate, etc?",
            input_variables=["industry_title"]
        ),
        None,
        "Influential economic factors"
    ),
    (
        PromptTemplate(
            template="What are the legal or regulatory factors affecting the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Influential legal factors"
    ),
    (
        PromptTemplate(
            template="How do political factors influence the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Influential political factors"
    ),
    (
        PromptTemplate(
            template="How is technology influencing the {industry_title} industry?",
            input_variables=["industry_title"]
        ),
        None,
        "Influential technological factors"
    ),
    (
        PromptTemplate(
            template="How does the {industry_title} industry affect the environment "
                     "and how is it addressing these impacts?",
            input_variables=["industry_title"]
        ),
        None,
        "Effects on the environments"
    )
]


human_prompts, OUTPUT_PARSERS, HEADINGS = list(zip(*human_prompts_and_parsers_and_headings))

PROMPTS = [
    ChatPromptTemplate.from_messages([
        system_msg_prompt,
        HumanMessagePromptTemplate(prompt=prompt)
    ])
    for prompt in human_prompts
]

MULTIPLES_TBL_PROMPT = ChatPromptTemplate.from_messages([
    system_msg_prompt,
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Examine the provided multiples table, which offers a comprehensive overview of key valuation "
                     "multiples pertinent to a specific industry. "
                     "Your elucidation should encompass a minimum of three to four well-structured paragraphs. "
                     "In your analysis, expound upon each entry within the table, presenting factual insights that "
                     "shed light on the significance and implications of these valuation multiples within the industry "
                     "context. Furthermore, where applicable, incorporate pertinent definitions and supplementary "
                     "information to enhance the reader's comprehension of these valuation metrics and their relevance "
                     "in evaluating industry performance and investment opportunities."
                     "\n\n>>>\n{multiples_table}\n<<<\n\nYOUR RESPONSE:",
            input_variables=["multiples_table"]
        )
    )
])


RISK_TBL_PROMPT = ChatPromptTemplate.from_messages([
    system_msg_prompt,
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Analyze the presented table showcasing the risk components associated with a specific industry. "
                     "Your analysis should be detailed and comprehensive, spanning at least four paragraphs. "
                     "As you delve into each element of the table, provide clear explanations elucidating the "
                     "significance and impact of these risk components on the industry. "
                     "Additionally, where appropriate, furnish relevant insights, definitions, or contextual "
                     "information that can contribute to a more profound grasp of these risk factors."
                     "\n\n>>>\n{risk_table}\n<<<\n\nYOUR RESPONSE:",
            input_variables=["risk_table"]
        )
    )
])


BARRIERS_TO_ENTRY_PROMPT = ChatPromptTemplate.from_messages([
    system_msg_prompt,
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=
            "Analyze the provided dataset outlining the checklist of barriers to entry within a specific industry. "
            "Elaborate on each element of the checklist, providing comprehensive explanations and, where applicable, "
            "offer relevant information or definitions that can aid in better comprehending these barriers."
            "\n\n>>>\n{barriers_to_entry}\n<<<\n\nYOUR RESPONSE:",
            input_variables=["barriers_to_entry"]
        )
    )
])
