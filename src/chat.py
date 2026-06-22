import os
import sys

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main():
    try:
        chain = search_prompt()
    except Exception as error:  # noqa: BLE001
        print(f"Não foi possível iniciar o chat: {error}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite 'sair' para encerrar.\n")

    while True:
        try:
            question = input("Faça sua pergunta:\n\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando o chat.")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando o chat.")
            break

        try:
            answer = chain.invoke(question)
        except KeyboardInterrupt:
            print("\nEncerrando o chat.")
            break
        except Exception as error:  # noqa: BLE001
            print(f"RESPOSTA: Ocorreu um erro ao processar sua pergunta: {error}\n")
            continue

        print(f"RESPOSTA: {answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
    # The Gemini gRPC client prints a harmless traceback during the normal
    # interpreter shutdown (its atexit finalizer runs after gRPC's own globals
    # are gone). Flush our output and exit immediately to skip that teardown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
