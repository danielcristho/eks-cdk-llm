import aws_cdk as core
import aws_cdk.assertions as assertions

from scale_llm.scale_llm_stack import ScaleLlmStack

# example tests. To run these tests, uncomment this file along with the example
# resource in scale_llm/scale_llm_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = ScaleLlmStack(app, "scale-llm")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
