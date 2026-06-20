"""
Setup RVC — Instalação e Download de Modelos de Voz
=====================================================
Este script ajuda a:
  1. Instalar as dependências do RVC (rvc-python, huggingface_hub, etc.)
  2. Procurar e descarregar modelos de vozes famosas do HuggingFace
  3. Configurar o ambiente para voice cloning em tempo real

Uso:
  python setup_rvc.py                  # Menu interativo
  python setup_rvc.py install          # Instalar dependências
  python setup_rvc.py download Voz     # Descarregar modelo específico
  python setup_rvc.py list             # Listar modelos disponíveis
"""

import os
import sys
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Cores para terminal
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def print_ok(text: str):
    print(f"{GREEN}✅ {text}{RESET}")


def print_warn(text: str):
    print(f"{YELLOW}⚠️  {text}{RESET}")


def print_err(text: str):
    print(f"{RED}❌ {text}{RESET}")


# ---------------------------------------------------------------------------
# Passo 1: Instalar dependências
# ---------------------------------------------------------------------------
def install_dependencies():
    """Instala todas as dependências necessárias para voice cloning."""
    print_header("Passo 1: Instalar dependências")

    packages = [
        "rvc-python",
        "huggingface_hub",
        "pyaudio",
        "numpy",
        "soundfile",
        "librosa",
        "torch",
        "torchaudio",
    ]

    print("Pacotes a instalar:")
    for pkg in packages:
        print(f"  📦 {pkg}")

    print(f"\n{YELLOW}Nota: rvc-python pode demorar uns minutos a compilar.{RESET}")
    resp = input("Continuar? [S/n]: ").strip().lower()
    if resp and resp != "s":
        print("Cancelado.")
        return False

    for pkg in packages:
        print(f"\nA instalar {pkg}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
            )
            print_ok(f"{pkg} instalado")
        except subprocess.CalledProcessError:
            print_warn(f"Falha ao instalar {pkg} — tentando continuar...")

    # Verificar se rvc-python ficou funcional
    try:
        import rvc_python
        print_ok("rvc-python importado com sucesso!")
    except ImportError:
        print_err("rvc-python NÃO foi instalado corretamente.")
        print("  Tenta manualmente: pip install rvc-python")
        print("  Ou visita: https://github.com/ddPn08/rvc-python")

    print("\nInstalação concluída!")
    return True


# ---------------------------------------------------------------------------
# Passo 2: Descarregar modelos de voz
# ---------------------------------------------------------------------------
VOZES_CONHECIDAS = {
    "taylor_swift": {
        "descricao": "Taylor Swift (voz feminina pop)",
        "tags": ["taylor", "swift", "female", "pop", "singer"],
    },
    "donald_trump": {
        "descricao": "Donald Trump (voz masculina grave)",
        "tags": ["trump", "donald", "male", "deep", "politician"],
    },
    "morgan_freeman": {
        "descricao": "Morgan Freeman (voz masculina narrador)",
        "tags": ["morgan", "freeman", "male", "narrator", "deep"],
    },
    "obama": {
        "descricao": "Barack Obama (voz masculina política)",
        "tags": ["obama", "barack", "male", "politician"],
    },
    "elon_musk": {
        "descricao": "Elon Musk (voz masculina tech)",
        "tags": ["elon", "musk", "male", "tech"],
    },
    "biden": {
        "descricao": "Joe Biden (voz masculina política)",
        "tags": ["biden", "joe", "male", "politician"],
    },
    "kanye_west": {
        "descricao": "Kanye West (voz masculina rap)",
        "tags": ["kanye", "west", "male", "rap", "hiphop"],
    },
    "eminem": {
        "descricao": "Eminem (voz masculina rap rápido)",
        "tags": ["eminem", "male", "rap", "fast"],
    },
    "ariana_grande": {
        "descricao": "Ariana Grande (voz feminina pop)",
        "tags": ["ariana", "grande", "female", "pop", "high"],
    },
    "lady_gaga": {
        "descricao": "Lady Gaga (voz feminina versátil)",
        "tags": ["lady", "gaga", "female", "pop", "versatile"],
    },
    "drake": {
        "descricao": "Drake (voz masculina R&B/rap)",
        "tags": ["drake", "male", "rap", "rnb"],
    },
    "narrador_br": {
        "descricao": "Narrador Brasileiro (voz grave locução)",
        "tags": ["narrador", "brasileiro", "male", "deep", "locucao"],
    },
    "galvao_bueno": {
        "descricao": "Galvão Bueno (narrador esportivo BR)",
        "tags": ["galvao", "bueno", "narrador", "esportivo", "brasileiro"],
    },
}

REPOSITORIOS_RVC = [
    "therealvulcan/RVC-Voice-Conversion-Community-Models",
    "lj1995/VoiceConversionWebUI",
    "Jeffrey2304/RVCModels",
    "QuickWick/RVCModels",
]


def search_and_download(voice_query: str):
    """Procura e descarrega um modelo RVC de voz."""
    print_header(f"Procurar modelo: {voice_query}")

    os.makedirs("models/rvc", exist_ok=True)

    # Verificar se já existe
    existing = list(Path("models/rvc").glob("*.pth"))
    for f in existing:
        if voice_query.lower() in f.stem.lower():
            print_ok(f"Modelo já existe: {f.name}")
            return str(f).replace(".pth", "")

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print_err("huggingface_hub não instalado. Corre: python setup_rvc.py install")
        return None

    query_lower = voice_query.lower()

    for repo in REPOSITORIOS_RVC:
        try:
            print(f"  Procurando em {repo}...")
            files = list_repo_files(repo)
            
            # Procurar .pth
            pth_files = [
                f for f in files
                if f.endswith(".pth")
                and any(tag in f.lower() for tag in query_lower.replace(" ", "_").split("_"))
            ]

            # Se não encontrou, tentar com o nome exato
            if not pth_files:
                pth_files = [
                    f for f in files
                    if f.endswith(".pth") and query_lower.replace(" ", "_") in f.lower()
                ]

            if pth_files:
                pth_file = pth_files[0]
                print(f"    Encontrado: {pth_file}")

                # Descarregar .pth
                local_pth = hf_hub_download(
                    repo, pth_file,
                    local_dir="models/rvc",
                    local_dir_use_symlinks=False,
                )
                print_ok(f"Descarregado: {os.path.basename(local_pth)}")

                # Descarregar .index se existir
                index_file = pth_file.replace(".pth", ".index")
                if index_file in files:
                    local_idx = hf_hub_download(
                        repo, index_file,
                        local_dir="models/rvc",
                        local_dir_use_symlinks=False,
                    )
                    print_ok(f"Index: {os.path.basename(local_idx)}")

                return local_pth.replace(".pth", "")

        except Exception as e:
            print(f"    Erro: {e}")
            continue

    print_warn(f"Nenhum modelo encontrado para '{voice_query}'")
    print("\nSugestões:")
    print("  1. Visita: https://huggingface.co/models?search=rvc+voice+conversion")
    print("  2. Procura no Discord do RVC: https://discord.gg/aihub")
    print("  3. Coloca manualmente o .pth e .index em models/rvc/")
    return None


# ---------------------------------------------------------------------------
# Menu interativo
# ---------------------------------------------------------------------------
def interactive_menu():
    """Menu interativo principal."""
    print_header("🎤 Setup RVC — Voice Cloning para Deep-LiveCam")
    print("Este assistente vai ajudar-te a configurar voice cloning.\n")

    while True:
        print(f"\n{BOLD}Opções:{RESET}")
        print("  1. Instalar dependências (rvc-python, torch, etc.)")
        print("  2. Procurar e descarregar voz famosa")
        print("  3. Listar vozes conhecidas")
        print("  4. Ver modelos já descarregados")
        print("  5. Testar voice cloner")
        print("  0. Sair")

        choice = input("\nEscolha: ").strip()

        if choice == "1":
            install_dependencies()

        elif choice == "2":
            print("\nVozes famosas disponíveis:")
            for i, (key, info) in enumerate(VOZES_CONHECIDAS.items()):
                print(f"  {i+1:2d}. {key:<20s} — {info['descricao']}")
            
            query = input("\nNome da voz (ex: taylor_swift) ou pesquisa livre: ").strip()
            if query:
                # Verificar se é uma chave conhecida
                if query.lower() in VOZES_CONHECIDAS:
                    info = VOZES_CONHECIDAS[query.lower()]
                    print(f"\n  {info['descricao']}")
                
                result = search_and_download(query)
                if result:
                    print(f"\n{ GREEN}✅ Pronto! Modelo em: {result}{RESET}")
                    print(f"  Para usar: python run_voice_avatar.py --voice-model {result}")

        elif choice == "3":
            print_header("Vozes Conhecidas")
            print(f"{'Nome':<22s} {'Descrição'}")
            print("-" * 60)
            for key, info in VOZES_CONHECIDAS.items():
                print(f"  {key:<20s} — {info['descricao']}")
            print(f"\n{YELLOW}Nota: A disponibilidade depende do HuggingFace.{RESET}")
            print("Podes também procurar qualquer voz manualmente.")

        elif choice == "4":
            print_header("Modelos Descarregados")
            rvc_dir = Path("models/rvc")
            if rvc_dir.exists():
                pth_files = list(rvc_dir.glob("*.pth"))
                index_files = list(rvc_dir.glob("*.index"))
                if pth_files:
                    for pth in pth_files:
                        idx = pth.with_suffix(".index")
                        has_idx = "✅" if idx.exists() else "⚠️ (sem index)"
                        size_mb = pth.stat().st_size / 1024 / 1024
                        print(f"  🎤 {pth.stem:<30s} {size_mb:.1f}MB {has_idx}")
                else:
                    print("  Nenhum modelo encontrado em models/rvc/")
            else:
                print("  Pasta models/rvc/ não existe. Corre opção 2 primeiro.")

        elif choice == "5":
            print_header("Testar Voice Cloner")
            try:
                from voice_cloner import VoiceCloner
                cloner = VoiceCloner()
                models = cloner.list_available_models()

                valid_models = [m for m in models if m.is_valid]
                if valid_models:
                    print("Modelos disponíveis:")
                    for i, m in enumerate(valid_models):
                        print(f"  [{i}] {m.name}")
                    
                    idx = input("\nEscolhe o modelo (Enter=primeiro): ").strip()
                    idx = int(idx) if idx.isdigit() else 0
                    if 0 <= idx < len(valid_models):
                        m = valid_models[idx]
                        cloner.load_model(str(m.pth_path).replace(".pth", ""))
                        print(f"\nCarregado: {m.name}")
                        print("Pressiona Ctrl+C para parar o teste.")
                        cloner.start()
                        try:
                            import time
                            while True:
                                time.sleep(0.1)
                        except KeyboardInterrupt:
                            cloner.stop()
                            print("\nTeste concluído!")
                else:
                    print_warn("Nenhum modelo .pth disponível.")
                    print("Usando efeitos simples...")
                    cloner.start()
                    try:
                        import time
                        while True:
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        cloner.stop()
                        print("\nTeste concluído!")

            except ImportError as e:
                print_err(f"Erro: {e}")
                print("Corre opção 1 primeiro.")

        elif choice == "0":
            print("\n👋 Até logo!")
            break

        else:
            print_warn("Opção inválida!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == "install":
            install_dependencies()
        elif cmd == "download":
            if len(sys.argv) > 2:
                voice = sys.argv[2]
                search_and_download(voice)
            else:
                print("Uso: python setup_rvc.py download <nome_da_voz>")
                print("Ex:   python setup_rvc.py download taylor_swift")
        elif cmd == "list":
            print_header("Vozes Conhecidas")
            for key, info in VOZES_CONHECIDAS.items():
                print(f"  {key:<20s} — {info['descricao']}")
        elif cmd == "installed":
            rvc_dir = Path("models/rvc")
            if rvc_dir.exists():
                for pth in rvc_dir.glob("*.pth"):
                    size_mb = pth.stat().st_size / 1024 / 1024
                    print(f"  {pth.stem} ({size_mb:.1f}MB)")
            else:
                print("Nenhum modelo instalado.")
        else:
            print(f"Comando desconhecido: {cmd}")
            print("Comandos: install, download <voz>, list, installed")
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
