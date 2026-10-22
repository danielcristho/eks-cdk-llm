import aws_cdk as core
import aws_cdk.assertions as assertions

from pipeline_llm.pipeline_llm_stack import PipelineLlmStack

# example tests. To run these tests, uncomment this file along with the example
# resource in pipeline_llm/pipeline_llm_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = PipelineLlmStack(app, "pipeline-llm")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
