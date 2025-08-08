import os, json, time
import sys
import csv
import pandas as pd
from datetime import datetime

from flowcept.agents import build_llm_model
from flowcept.agents.agent_client import run_tool
from langchain_community.llms.sambanova import SambaStudio
from transformers import AutoTokenizer
# SETUP FIXED PARAMETERS FOR THE CSV LABELING 
# MODEL = "8B" # Input your model name here. We tested with: GPT,70B,8B,gemini,claude
# CONFIG = "HC" # Input your configuration for Temp and Top_P here if used and have a name for it, if not you can leave it empty or tag it out.
# RAG = "F" # This is your context / RAG Type. Same thing. Ex) 'F' for Full context. 'BL' for baseline. The actual ones are in the paper.
# WFE = "100"  # Number of workflow executions.
# REP = "0"
# JUDGE = "GPT"

MODEL = sys.argv[1]  # e.g., "gpt" or "70B" or "8B"
CONFIG = sys.argv[2]  # e.g., "HC" or ""
RAG = sys.argv[3]     # e.g., "F", "BL"
WFE = sys.argv[4]     # e.g., "100"
REP = sys.argv[5]  # e.g. 0
JUDGE = sys.argv[6]  # e.g., GPT

Model_Config = f"{MODEL}|{CONFIG}|{RAG}|{WFE}|{REP}|{JUDGE}"
print(f"Model_Config = {Model_Config}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# QUERIES 
queries = [

    {
        "id": 1,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["ControlFlow","Scheduling"],
            "prov_type": "Retrospective"
        },
        "query": "For each workflow execution, for each hostname, report the number of tasks executed.",
        "expected_response": {  
            "code": "result = df.groupby(['workflow_id', 'hostname']).size().reset_index(name='count')",
            "fields_in_code":["workflow_id","started_at"],
            "query_constructs_in_code":["groupby|agg","min|sort","count|size|length"],
              "regex_to_contain_match": [
            r"workflow_id.*task_count",  # CSV header
            r"[a-f0-9\-]{8,}.*\d+",     # UUID and count data
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"  # Timestamp format
        ], 
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 2, 
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["DataFlow"],
            "prov_type": "Retrospective"
        },
        "query": "For each activity in the last workflow executed, what is the average output?",
        "expected_response": {
            "code":"result = df[df.workflow_id == df.loc[df.ended_at.idxmax(), 'workflow_id']].groupby('activity_id').agg({col: 'mean' for col in df.columns if col.startswith('generated.')})",
            "fields_in_code": ["activity_id","generated."],  
            "query_constructs_in_code":["groupby|agg","mean|avg"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 3,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "What is the execution time per task in the first workflow execution?",
        "expected_response": {
            "code":"result = df[df.workflow_id == df.loc[df.started_at.idxmin(), 'workflow_id']][['task_id', 'telemetry_summary.duration_sec']]",
            "fields_in_code": ["workflow_id","task_id","telemetry_summary.duration_sec"],
            "query_constructs_in_code":["sorted|sort"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 4,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "What is the duration of each workflow execution?",
        "expected_response": {
            "code":"result=df.groupby('workflow_id').agg(first=('started_at','min'), last=('ended_at','max')).reset_index().assign(duration=lambda x: x['last'] - x['first'])[['workflow_id', 'duration']]",
            "fields_in_code": ["started_at","ended_at","duration","workflow_id","activity_id"],
            "query_constructs_in_code":["pd.to_datetime","groupby","sum","loc","idxmax|max"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 5,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "What is the longest workflow execution?",
        "expected_response": {
            "code":"result = df.groupby('workflow_id').agg(first=('started_at','min'), last=('ended_at','max')).reset_index().assign(duration=lambda x: x['last'] - x['first'])[['workflow_id', 'duration']].sort_values(by='duration', ascending=False).head(1)['workflow_id']",
            "fields_in_code": ["started_at","ended_at","workflow_id"],
            "query_constructs_in_code":["pd.to_datetime","groupby|agg","max","idxmax"],
            "number_of_rows":"one",
            "number_of_columns":"one"
        }
    },
    {
        "id": 6,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["Scheduling"],
            "prov_type": "Retrospective"
        },
        "query": "What node is most often used?",
        "expected_response": {
            "code":"result = df['hostname'].value_counts().idxmax()",
            "fields_in_code": ["hostname"],
            "query_constructs_in_code":["value_counts|count"],
            "number_of_rows":"one",
            "number_of_columns":"one"
        }
    },
    {
        "id": 7,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "What is the longest task, and what is it's duration?",
        "expected_response": {
            "code":"result = df.loc[df['telemetry_summary.duration_sec'].idxmax(), ['task_id', 'telemetry_summary.duration_sec']]",
            "fields_in_code": ["telemetry_summary.duration_sec", "task_id"],
            "query_constructs_in_code":["max"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 8,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["DataFlow", "ControlFlow"],
            "prov_type": "Prospective"
        },
        "query": "List the unique activities in all workflows",
        "expected_response": {
            "code": "result = df['activity_id'].unique()",
            "fields_in_code": ["activity_id"],
            "query_constructs_in_code":["unique"],
            "number_of_rows":"one",
            "number_of_columns":"one"
        }
    },
    {
        "id": 9,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["DataFlow", "ControlFlow"],
            "prov_type": "Retrospective"
        },
        "query": "What is the input and output value of the first task executed in the first workflow execution?",
        "expected_response": {
            "code":"result = df.loc[df['started_at'].idxmin()][[c for c in df.columns if c.startswith('used.') or c.startswith('generated.')]].dropna()",
            "fields_in_code": ["started_at","activity_id","used.input_value","generated.h"],
            "query_constructs_in_code":["pd.to_datetime"],
            "number_of_rows":"one",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 10,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["Scheduling", "Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "For the most recent workflow execution, which node did each task run on and how long each task take?",
        "expected_response": {
            "code":"result = df[df.workflow_id == df.loc[df.ended_at.idxmax(), 'workflow_id']][['task_id', 'hostname', 'telemetry_summary.duration_sec']]",
            "fields_in_code": ["workflow_id","started_at","hostname","task_id","telemetry_summary.duration_sec"],
            "query_constructs_in_code":["loc","pd.to_datetime","idxmax","sort_values"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 11,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["DataFlow", "ControlFlow", "Scheduling", "Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "Provide a summary of all activities, including start times, durations, execution nodes, and output values.",
        "expected_response": {
            "code":"result = df.groupby('activity_id').agg({**{'started_at': ['min', 'max'], 'telemetry_summary.duration_sec': ['max', 'min', 'mean']}, **{col: ['max', 'min', 'mean'] for col in df.columns if col.startswith('generated.')}})",
            "fields_in_code": ["activity_id","started_at","telemetry_summary.duration_sec","hostname","generated."],
            "query_constructs_in_code":["groupby|agg|sortvalues","min","max","mean"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 12,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["DataFlow", "ControlFlow", "Scheduling", "Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "For the first workflow execution, list all tasks with their start times, durations, nodes, and output values.",
        "expected_response": {
            "code":"result = df[df['workflow_id'] == sorted(df['workflow_id'].unique())[0]][['activity_id','started_at','telemetry_summary.duration_sec','hostname'] + [col for col in df.columns if col.startswith('generated.')]].sort_values('started_at')",
            "fields_in_code": ["workflow_id","activity_id","started_at","telemetry_summary.duration_sec","hostname","generated."],
            "query_constructs_in_code":["sorted","unique","sort_values"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 13,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["ControlFlow", "Scheduling"],
            "prov_type": "Retrospective"
        },
        "query": "For each workflow, list the unique tasks executed in the order they started",
        "expected_response": {
            "code":"result = df.sort_values(['workflow_id', 'started_at']).groupby('workflow_id')['task_id'].unique()",
            "fields_in_code": ["workflow_id","started_at","task_id","campaign_id"],
            "query_constructs_in_code":["sort_values","groupby"],
            "number_of_rows":"one",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 14, 
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["DataFlow"],
            "prov_type": "Retrospective"
        },
        "query": "Looking at all workflow executions, what is the minimum output value associated with each activity?",
        "expected_response": {
            "code":"result = df.groupby('activity_id')[[col for col in df.columns if col.startswith('generated.')]].min()",
            "fields_in_code": ["activity_id","generated."],
            "query_constructs_in_code":["groupby","apply","lambda","min","dropna"],
            "number_of_rows":"multiple",
            "number_of_columns":"one"
        }
    },
    {
        "id": 15,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["Scheduling"],
            "prov_type": "Retrospective"
        },
        "query": "For the first workflow, which node did each task start on?",
        "expected_response": {
            "code":"result = df[df.workflow_id == df.loc[df.started_at.idxmin(), 'workflow_id']][['task_id', 'hostname']]",
            "fields_in_code": ["workflow_id","task_id","hostname"],
            "query_constructs_in_code":["sorted","unique","drop_duplicates"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"
        }
    },
    {
        "id": 16,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["ControlFlow"],
            "prov_type": "Retrospective"
        },
        "query": "For each workflow, what is the order the activities started?",
        "expected_response": {
            "code":"result = df.groupby(['workflow_id', 'activity_id'], as_index=False).agg({'started_at': 'min'}).sort_values(by='started_at')",
            "fields_in_code": ["workflow_id","activity_id","used.*"],  
            "query_constructs_in_code":["list comprehension","shift","==","|","any","loc"], 
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"  # Number of activities can vary, so I am using 'variable' to indicate that the number of columns is not fixed.
        }
    },
    {
        "id": 17,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLAP",
            "data_type": ["Scheduling"],
            "prov_type": "Retrospective"
        },
        "query": "Which node executed the most tasks in the latest workflow run?",
        "expected_response": {
            "code": "result = df[df['workflow_id'] == df[df.started_at == df['started_at'].max()]['workflow_id'].values[0]].groupby('hostname', as_index=False).size().sort_values(by='size').head(1)['hostname']",  
            "fields_in_code": ["hostname"],
            "query_constructs_in_code":["value_counts","idxmax"],
            "number_of_rows":"one",
            "number_of_columns":"one"
        }
    },
    {
        "id": 18,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["DataFlow"],
            "prov_type": "Retrospective"
        },
        "query": "In the last 5 workflow executions, what is the largest generated value in each activity",
        "expected_response": {
            "code":"result = df[df.workflow_id.isin(df.groupby('workflow_id', as_index=False).agg({'ended_at': 'max'}).sort_values(by='ended_at', ascending=False).head(5)['workflow_id'])].groupby('activity_id').agg({c: 'max' for c in df.columns if c.startswith('generated.')})",
            "fields_in_code": ["workflow_id","activity_id","used.*"],
            "query_constructs_in_code":["sorted","unique","isin","groupby","max","startswith"],
            "number_of_rows":"multiple",
            "number_of_columns":"multiple"  # Number of activities can vary, so I am using 'variable' to indicate that the number of columns is not fixed.
        }
    },
    {
        "id": 19,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "List all workflows that took longer than 0.01 seconds to execute.",
        "expected_response": {
            "code":"result = df.groupby('workflow_id').filter(lambda x: (x['ended_at'].max() - x['started_at'].min()).total_seconds() > 0.01)['workflow_id'].unique().tolist()",
            "fields_in_code": ["started_at","ended_at","workflow_id"],
            "query_constructs_in_code":["assign","pd.to_datetime","groupby","filter","lambda","total_seconds","unique","tolist"],
            "number_of_rows":"multiple",
            "number_of_columns":"one"
        }
    },
    {
        "id": 20,
        "classification": {
            "who": "Human",
            "when": "Online",
            "scope": "Targeted Query",
            "query_type": "OLTP",
            "data_type": ["Telemetry"],
            "prov_type": "Retrospective"
        },
        "query": "For the last 10 workflow executions, what was the average percentage of cpu used per workflow execution.",
        "expected_response": {
            "code":"""result = df.groupby('workflow_id', as_index=False).agg({"ended_at": 'max',"telemetry_summary.cpu.percent_all_diff": "mean"}).sort_values(by='ended_at', ascending=False).head(10)[['workflow_id',"telemetry_summary.cpu.percent_all_diff"]]""",
            "fields_in_code": ["workflow_id","telemetry_summary.cpu.percent_all_diff"],
            "query_constructs_in_code":["isin","pd..unique","notna","mean"],
            "number_of_rows":"one",
            "number_of_columns":"one"
        }
    }
]
# SETUP INPUTS AND PATHS
input_dir = "/tmp"
df_path = f"{input_dir}/current_agent_df.csv"
schema_path = f"{input_dir}/current_tasks_schema.json"
value_examples_path = f"{input_dir}/value_examples.json"
df_workflow = pd.read_csv(df_path)
with open(schema_path, 'r') as f:
    dynamic_schema = json.dumps(json.load(f), indent=2)

with open(value_examples_path, 'r') as f:
    example_values = json.dumps(json.load(f), indent=2)


def build_judge_llm(model_kwargs=None, service_provider=None):
    """
    Build and return an LLM instance using agent configuration.

    This function retrieves the model name and keyword arguments from the AGENT configuration,
    constructs a SambaStudio LLM instance, and returns it.

    Returns
    -------
    LLM
        An initialized LLM object configured using the `AGENT` settings.
    """
    if service_provider == "sambanova":
        from langchain_community.llms.sambanova import SambaStudio
        assert os.environ.get("SAMBASTUDIO_URL", None) is not None
        assert os.environ.get("SAMBASTUDIO_API_KEY", None) is not None
        llm = SambaStudio(model_kwargs=model_kwargs)
    elif service_provider == "azure":
        from langchain_openai.chat_models.azure import AzureChatOpenAI
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        service_url = os.environ.get("AZURE_OPENAI_API_ENDPOINT", None)
        llm = AzureChatOpenAI(
            azure_deployment=model_kwargs.get("model"),
            azure_endpoint=service_url,

            api_key=api_key,
            **model_kwargs
        )
    elif service_provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY", None)
        llm = ChatOpenAI(openai_api_key=api_key, **model_kwargs)
    elif service_provider == "google":
        if "claude" in model_kwargs["model"]:
            api_key = os.environ.get("GOOGLE_API_KEY")
            model_kwargs["model_id"] = model_kwargs.pop("model")
            model_kwargs["google_token_auth"] = api_key
            from flowcept.agents.llms.claude_gcp import ClaudeOnGCPLLM
            llm = ClaudeOnGCPLLM(**model_kwargs)
        elif "gemini" in model_kwargs["model"]:
            from flowcept.agents.llms.gemini25 import Gemini25LLM
            llm = Gemini25LLM(**model_kwargs)
    return llm
def get_llm_model():
    # GPT:

    model_kwargs = {
        "model": "gpt-4",
        "temperature": 0.0,
        "top_p": 1.0,
        "api_version": "2023-05-15"
    }

    # Claude:
    # model_kwargs = {
    #     "model": 'claude-opus-4',
    #     "project_id": "ornl-frlow-wrkfeco-ai-proj",
    #     "anthropic_version": "vertex-2023-10-16",
    #     "location": "us-east5",
    #     "temperature": 0.00,
    #     "top_p": 1.0,
    # }
    llm = build_judge_llm(model_kwargs=model_kwargs, service_provider="azure")
    return llm

llm = get_llm_model()
# L_SCORE PROMPT
EVAL_PROMPT = '''You are an expert evaluator assessing whether a Python code snippet generated by an AI assistant to query a DataFrame `df` correctly answers a user's data analysis query.

The dataframe contains workflow provenance data from recent workflow executions. 
Here is the schema of common task fields in the DataFrame you will see:

    | Column                        | Data Type | Description |
    |-------------------------------|-------------|
    | `workflow_id`                 | string | Workflow the task belongs to. Use this field when the query is asking about workflow execution |
    | `task_id`                     | string | Task identifier. |
    | `parent_task_id`              | string | A task may be directly linked to others. Use this field when the query asks for a task informed by (or associated with or linked to) other task.  |
    | `activity_id`                 | string | Type of task (e.g., 'choose_option'). Use this for "task type" queries. One activity_id is linked to multiple task_ids. |
    | `campaign_id`                 | string | A group of workflows. |
    | `hostname`                    | string | Compute node name. |
    | `agent_id`                    | string | Set if executed by an agent. |
    | `started_at`                  | datetime64[ns, UTC] | Start time of a task. Always use this field when the query is has any temporal reference related to the workflow execution, such as 'get the first 10 workflow executions' or 'the last workflow execution'. |
    | `ended_at`                    | datetime64[ns, UTC] | End time of a task. | 
    | `subtype`                     | string | Subtype of a task. |
    | `tags`                        | List[str] | List of descriptive tags. |
    | `telemetry_summary.duration_sec` | float | Task duration (seconds) |
    | `telemetry_summary.cpu.percent_all_diff` | float | Difference in overall CPU utilization percentage across all cores between task end and start.|
    | `telemetry_summary.cpu.user_time_diff`   | float |  Difference average per core CPU user time ( seconds ) between task start and end times.|
    | `telemetry_summary.cpu.system_time_diff` | float |  Difference in CPU system (kernel) time (seconds) used during the task execution.|
    | `telemetry_summary.cpu.idle_time_diff`   | float |  Difference in CPU idle time (seconds) during task end and start.|
    ---
    For any queries involving CPU, use fields that begin with telemetry_summary.cpu

    These are the instructions that the AI assistant is aware of:

        -THERE ARE NOT INDIVIDUAL FIELDS NAMED `used` OR `generated`, they are ONLY are prefixes to the field names.
        -THERE IS NOT A FIELD NAMED `telemetry_summary.start_time` or `telemetry_summary.end_time` or `used.start_time` or `used.end_time`. Use `started_at` and `ended_at` instead when you want to find the duration of a task, activity, or workflow execution.
        -THE GENERATED FIELDS ARE LABELED AS SUCH: `generated.()` NOT `generated_output`. Any reference to `generated_output` is incorrect and should be replaced with `generated.` prefix.
        -THERE IS NOT A FIELD NAMED `execution_id` or `used.execution_id`. Look at the QUERY to decide what correct _id field to use. Any mentions of workflow use `workflow_id`. Any mentions of task use `task_id`. Any mentions of activity use `activity_id`.
        -DO NOT USE `nlargest` in the query code, use `sort_values` instead. The `nlargest` method is not supported by the DataFrame used in this workflow.
        -An activity with a value in the `generated.` column created that value. Whereas an activity that has a value in the `used.` column used that value from another activity. IF THE `used.` and `generated.` fields share the same letter after the dot, that means that the activity associated with the `generated.` was created by another activity and the one with `used.` used that SAME value that was created by the activity with that same value in the `generated.` field.
        -WHEN calculating total time of a workflow execution (identified by `workflow_id`), get its latest task's `ended_at` and its earliest task's `started_at`and compute the difference between them.
        -WHEN user requests duration or execution time per task or for individual tasks, utilize `telemetry_summary.duration_sec`. 
        -WHEN the user requests the first or last number of workflow executions, USE the `started_at` or `ended_at` field to sort the DataFrame and select the first or last rows accordingly.
        -WHEN the user requests the "first workflow", you must identify the workflow by using workflow_id of the task with the earliest started_at. DO NOT use the smallest workflow_id. To find "last workflow" use the latest started_at.
        -WHEN the user requests a "summary" of activities, you must incorporate relevant summary statistics such as min, max, and mean, into the code you generate.

Additionally, here is the activity-specific schema and example values available for your evaluation:

The schema for these fields is defined in the dictionary below.
It maps each activity ID to its inputs (i) and outputs (o), using flattened field names that include `used.` or `generated.` prefixes to indicate the role the field played in the task. These names match the columns in the dataframe `df`.

```python
{dynamic_schema}
Now, the other dictionary below contains example values for each field in the schema.
```python
{example_values}

YOU SHOULD USE THESE AS WELL AS THE COMMON TASK FIELDS TO EVALUATE THE CODE.

You are provided with:
1. The user's original query (natural language).
2. A gold standard code snippet that correctly answers the query.
3. The code generated by the AI assistant.

Compare the **logic or operations** used in both code snippets and assign a similarity score between 0.0 and 1.0 reflecting how functionally equivalent they are.

If you believe the generated dataframe query code fully and correctly answers the user's query, return a score of 1.0, even if it uses a different structure or approach than the gold standard code snippet, as long as the logic and operations are equivalent.

Return a 1.0 > score >= 0.8 if the generated code mostly responds to the user's query and is similar in functionality, even if there are differences in how it filters, groups, or returns the data. If the query output would be functionally equivalent to the gold snippet's, give it a score close to 1.0 (or 1.0), even if the logic differs. 

Return a 0.8 > score >= 0.5 if the generated code shares some similarities (such as similar constructs or partial logic) but is not fully functionally equivalent and would not return the same result or fully address the user's request.

Return a 0.5 > score >= 0.2 if the generated code tries to answer the user query, shares functionalities or logic comparing with the gold code snippet, shares some columns with the gold standard, but it has issues like different or missing or extra column names or applying a function to the wrong data type which would lead to an error, but the error could be easily fixed.

Return a 0.2 > score >= 0 if the generated code is entirely different from the gold standard code snippet, does not share any meaningful logic or operations, and does not address the user's query at all. It would require rewriting the query entirely from scratch to answer the user query.

In all cases, you must give higher scores based on whether the generated code from the AI assistant answers the USER'S QUERY EFFECTIVELY, meaning that the resulting data set would be similar to the one generated when running the gold standard code snippet, even if it uses a different approach than the gold standard or if it does not generate an entirely valid Python code, or with different column names, but these errors could be easily fixed.

Focus on: 
- Whether the assistant's code performs the **similar data selection, filtering, grouping, and transformation steps** as the gold standard code snippet.
- Whether it returns a **DataFrame with similar structure and contents** to the result set generated when running the gold standard code.
- Whether any differences would affect the **correctness or completeness** of the answer to the user's query.

Ignore differences in:
- Formatting, whitespace, or variable names.
- It is fine if there are output differences in the structure or data type of the output, as long as the contents respond the user query.
- It is fine if there are more or less columns, as long as the contents respond the user query.
- Syntax variations that do not affect functionality.
- Minor differences in column names would not significantly affect the score.
- Dropping columns with NaN values, should NOT affect the score, as long as the logic is similar and the code would return a valid DataFrame.

Your output needs to be as such:
Please provide **only a single float number between 0.0 and 1.0** and use your best judgement to work on the decimals, like a score of 0.78 or even 0.15. I want you to give me a score and A SHORT CONCISE reason for said score.
Your output format MUST be a VALID JSON as follows:
{{
"score": float number,
"reason": "reason"
}}
Please Respond ONLY in with a VALID JSON format which will be parsed. DO NOT include any special symbols like ```json or Markdown or any other marks. ONLY THE RAW VALID JSON!.
---
**User Query**:
{user_query}
**Gold Standard Code**:
```python
{expected_code}
```
**AI Assistant generated Code**:
```python
{generated_code}
```
'''
def evaluate_code_similarity(user_query, expected_code, generated_code):
    prompt = EVAL_PROMPT.format(user_query=user_query, expected_code=expected_code, generated_code=generated_code,dynamic_schema=dynamic_schema, example_values=example_values)
    response = llm.invoke(prompt)
    
    if not response:
        
        raise Exception("Empty response from LLM. This should never happen.")
    try:
         # Match score in the 0.0 to 1.0 range
        # score_match = re.search(r'\*\*Score:\s*(0(?:\.\d+)?|1(?:\.0+)?)\*\*', response)
        # reason_match = re.search(r'\*\*Reason:\*\*\s*(.+)', response, re.DOTALL)
        # if score_match:
        #     score = float(score_match.group(1))
        # else:
        #     print(f"[WARN] Could not find score. Response:\n{response}")
        #     return 0.0, "Could not find score in LLM response."
        # reason = reason_match.group(1).strip() if reason_match else "No reason found."
        # # Truncate to 250 tokens
        # reason_tokens = tokenizer.encode(reason, add_special_tokens=False)
        # if len(reason_tokens) > 250:
        #     reason = tokenizer.decode(reason_tokens[:250]).strip()
        try:
            r = response.content
        except:
            r = response
        print("Response:", r)
        resp_obj = json.loads(r)
        score, reason = resp_obj.get('score', 0.0), resp_obj.get('reason', "No reason provided.")
        return score, reason
    except Exception as e:
        print(f"[ERROR] Failed to parse:\n{response}\nError: {e}")
        return 0.0, "Could not parse L_reason from LLM response."
# TOKEN COUNT
def count_tokens(text):
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

# EVALUATION LOOP
results = []
for query in queries:
    query_id = query['id']
    query_text = query['query']
    expected_code = query['expected_response']['code']
    t0 = time.time()
    try:
        # if query_id != 19:
        #     continue
        agent_output = run_tool(
            tool_name="query_on_saved_df",
            kwargs={
                "query": query_text,
                "dynamic_schema_path": schema_path,
                "value_examples_path": value_examples_path,
                "df_path": df_path
            }
        )[0]
        t1 = time.time()
        time_taken = round(t1 - t0, 4)
        result_wrapper = json.loads(agent_output)

        status_code = result_wrapper.get("code", 301)
        if status_code == 301:
            generated_code = result_wrapper.get("result", None).get("result_code")
        elif status_code < 300:
            raise Exception("This is not expected")
        else:
            try:
                generated_code = result_wrapper.get("result", None).get("result_code")
            except:
                generated_code = result_wrapper.get("extra").get("generated_code")
        # Token & Output fields
        input_prompt = result_wrapper.get("extra").get("prompt")
        input_tokens = count_tokens(input_prompt)

        output_tokens = count_tokens(generated_code)


        summary_text = "" 
        result_df_text = result_wrapper.get("result_df", "") if isinstance(result_wrapper, dict) else ""
        fallback_text = agent_output if isinstance(agent_output, str) else json.dumps(agent_output)
        if not summary_text:
            summary_text = fallback_text
        if not result_df_text:
            result_df_text = ""
        overall_text = summary_text + ("\n" + result_df_text if result_df_text else "") + "\n\nFALLBACKTEXT:\n"+fallback_text
        summary_tokens = count_tokens(summary_text)
        overall_tokens = input_tokens + output_tokens
        # Evaluate L_score
        L_score, L_reason = evaluate_code_similarity(query_text, expected_code, generated_code)
        # Combined results row
        row = {
            'query_id': query_id,
            'query': query_text,
            'expected_code': expected_code,
            'generated_code': generated_code,
            "status": status_code,
            #'summary_text': summary_text,
            # 'result_df_text': result_df_text,
            'model': MODEL,
            'model_settings': CONFIG,
            'context': RAG,
            'number_of_workflows': WFE,
            "judge": JUDGE,
            "rep": REP,
            'Model_Config': Model_Config,
            #'summary_tokens': summary_tokens,
            'overall_tokens': overall_tokens,
            'time_taken_seconds': time_taken,
            # 'full_agent_output': agent_output,
            'L_score': L_score,
            'L_reason': L_reason,
            "prompt": input_prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            'query_type': query['classification']['query_type'],
            'data_type': '|'.join(query['classification']['data_type']),
            'prov_type': query['classification']['prov_type'],
        }
        results.append(row)
        print(f"Query {query_id} done. L_score: {L_score:.2f}")
    except Exception as e:
        print(f"Failed on query {query_id}: {e}")
        row = {
            'query_id': query_id,
            'query': query_text,
            'expected_code': expected_code,
            'generated_code': None,
            "status": "ERROR",
            #'summary_text': "",
            'result_df_text': "",
            'model': MODEL,
            'model_settings': CONFIG,
            'context': RAG,
            'number_of_workflows': WFE,
            "judge": JUDGE,
            "rep": REP,
            'Model_Config': Model_Config,
            #'summary_tokens': 0,
            'overall_tokens': 0,
            'time_taken_seconds': round(time.time() - t0, 4),
            #'full_agent_output': None,
            'L_score': -100,
            'L_reason': f"ERROR: {e}",
            'query_type': query['classification']['query_type'],
            'data_type': '|'.join(query['classification']['data_type']),
            'prov_type': query['classification']['prov_type'],
            'error': f"Failed during processing: {e}"
        }
        results.append(row)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


# FINALIZE RESULTS
df_results = pd.DataFrame(results)
print("\n--- Evaluation Summary ---")
print(f"L_score mean: {df_results['L_score'].mean():.4f}")
# SAVING
output_path = f"/tmp/final_agent_output_{Model_Config}.csv"
#df_results.to_csv(output_path, index=False)
df_results.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')

print(f"Final results written to {output_path}")