from http.client import HTTPException
import os
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request
from fastapi.security import APIKeyHeader
from langgraph_deploy_demo.state import State
from langgraph_deploy_demo.graph import builder
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from contextlib import asynccontextmanager

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database connections when the app starts
    async with AsyncPostgresStore.from_conn_string(
        os.environ["DATABASE_URL"]
    ) as store:
        async with AsyncPostgresSaver.from_conn_string(
            os.environ["DATABASE_URL"]
        ) as checkpointer:
            app.state.store = store
            app.state.checkpointer = checkpointer

            await app.state.store.setup()
            await app.state.checkpointer.setup()

            yield

            # Clean up connections when the app shuts down
            await app.state.store.close()
            await app.state.checkpointer.close()


app = FastAPI(lifespan=lifespan)


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


@app.post("/generate", dependencies=[Depends(api_key_auth)])
async def generate(state: State, request: Request):
    if not request.app.state.store or not request.app.state.checkpointer:
        raise HTTPException(
            status_code=500,
            detail="Database or checkpoint store not found. Please check your DATABASE_URL.",
        )
    graph = builder.compile(
        store=request.app.state.store,
        checkpointer=request.app.state.checkpointer,
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
