import aws_cdk as core
import aws_cdk.assertions as assertions

from hypertuning_llm.hypertuning_llm_stack import HypertuningLlmStack

def test_sqs_queue_created():
    app = core.App()
    stack = PipelineLlmStack(app, "pipeline-llm")
    template = assertions.Template.from_stack(stack)

    template.has_resource_properties("AWS::SQS::Queue", {
        "VisibilityTimeout": 300
    })
