# Building a Research Agent

## Initial Plan

Initial plan was to understand and make an agent with the basics of LangGraph framework, which includes the usage of components like states, nodes, graphs.

### Basic Architecture

![img](https://raw.githubusercontent.com/nakibworkspace/learning-agents/eec99615a28e8c8256aae7e99f609455fbac8a7b/assets/research_agent.svg)


### State/Reducers
In state, the prior messages get replaced by the the messages written by the consecutive nodes. To handle this, reducers are implemented. Reducers handles this issue by appending every messages to the state resulting a full stack of messages stored. It also handles deduplications, and no nodes can delete any messages by writing.

![img](reducers)

### Agent Loop
Agent loop is applied to the agent in order to gain best reasoning of the model. 1 turn might not be good enough for the agent to respond, for example the research might not be sufficient enough, or the summarization is not well off. It needs iteration, achieved through agent loop.

![img](loop)

