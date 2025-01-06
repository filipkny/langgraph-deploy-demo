from http.client import HTTPException
import os
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.security import APIKeyHeader
from src.memory_agent.state import State
from src.memory_agent.graph import builder
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

load_dotenv()

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


X_API_KEY = APIKeyHeader(name="X_API_KEY")


def api_key_auth(x_api_key: str = Depends(X_API_KEY)):
    """takes the X-API-Key header and validate it with the X-API-Key in the database/environment"""
    if x_api_key != os.environ["X_API_KEY"]:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key. Check that you are passing a 'X-API-Key' on your header.",
        )


# Add a route to run the ai agent
@app.post("/generate", dependencies=[Depends(api_key_auth)])
async def generate_route(state: State):
    async with AsyncPostgresStore.from_conn_string(
        os.environ["DATABASE_URL"]
    ) as store:
        await store.setup()
        async with AsyncPostgresSaver.from_conn_string(
            os.environ["DATABASE_URL"]
        ) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(
                store=store,
                checkpointer=checkpointer,
            )
            graph.name = "LangGraphDeployDemo"
            print(state)
            config = {"configurable": {"thread_id": "3"}}
            try:
                result = await graph.ainvoke(state, config)
                return {"success": True, "result": result}
            except Exception as e:
                print(e)
                raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
