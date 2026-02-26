class AgentiumError(Exception):
    pass

class UnknownRecipientError(AgentiumError):
    pass

class UnknownToolError(AgentiumError):
    pass

class InvalidLLMDecisionError(AgentiumError):
    pass

class InvalidMessageError(AgentiumError):
    pass

class InvalidToolOutputError(AgentiumError):
    pass

class InvalidToolInputError(AgentiumError):
    pass

class InvalidToolExecutionError(AgentiumError):
    pass

class InvalidAgentExecutionError(AgentiumError):
    pass



