# ================== 导入核心依赖 ==================
import random
import os
import json
import textwrap
import time
from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ================== 初始化大模型 ==================
load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()

# ================== 工具函数 ==================
def print_separator(char="=", width=55):
    print(char * width)

def print_title(text, char="=", width=55):
    print_separator(char, width)
    print(f"  {text}")
    print_separator(char, width)

def slow_print(text, delay=0.03):
    """逐字打印，增加临场感"""
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

# ================== 1. 定义游戏状态 ==================
class GameState(TypedDict):
    """游戏状态字典，存储整个游戏的所有关键数据"""
    civilian_word: str          # 平民词语
    undercover_word: str        # 卧底词语
    role_assignment: dict       # 角色分配：{player/agent1: ("平民"/"卧底", 词语), ...}
    speeches: dict              # 当前轮发言：{player/agent1: "发言内容", ...}
    history_speeches: List[Dict[str, str]]  # 历史发言列表
    speech_reasoning: dict      # 发言策略理由
    votes: dict                 # 当前轮投票
    vote_reasoning: dict        # 投票理由
    game_status: str            # 游戏状态：running / end
    winner: str                 # 获胜方：civilian / undercover
    eliminated: List[str]       # 被淘汰的玩家列表
    round: int                  # 当前游戏轮次
    # ---- 新增：玩家相关 ----
    player_name: str            # 真实玩家的名字
    player_role: str            # 玩家角色：平民 / 卧底
    player_word: str            # 玩家拿到的词语
    player_eliminated: bool     # 玩家是否已被淘汰
    all_players: List[str]      # 所有参与者列表（含玩家自己）

def init_game_state(player_name: str) -> GameState:
    return {
        "civilian_word": "",
        "undercover_word": "",
        "role_assignment": {},
        "speeches": {},
        "history_speeches": [],
        "speech_reasoning": {},
        "votes": {},
        "vote_reasoning": {},
        "game_status": "running",
        "winner": "",
        "eliminated": [],
        "round": 1,
        # 玩家相关初始化
        "player_name": player_name,
        "player_role": "",
        "player_word": "",
        "player_eliminated": False,
        "all_players": []
    }

# ================== 2. 节点函数 ==================

# ---- 节点1：生成词语 ----
def generate_words(state: GameState) -> GameState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的「谁是卧底」游戏出题人，需生成一组高质量的词语对。
核心要求：
1. 词语类型：日常物品/食品/场景（如：奶茶-果汁、牙刷-牙膏），避免生僻词
2. 语义关系：平民词与卧底词高度相似但核心特征不同，有足够博弈空间
3. 难度适配：适合5人游戏，既不轻易暴露也能通过描述区分
4. 输出格式：必须严格按照 JSON 格式输出，示例：{{"civilian": "奶茶", "undercover": "果汁"}}
禁止输出任何额外文字，只返回JSON字符串！"""),
        ("user", "生成一组符合要求的谁是卧底词语对")
    ])
    chain = prompt | llm | parser
    result = chain.invoke({})

    try:
        word_data = json.loads(result.strip())
        civilian_word = word_data["civilian"]
        undercover_word = word_data["undercover"]
    except (json.JSONDecodeError, KeyError):
        fallback_pairs = [
            ("奶茶", "果汁"), ("牙刷", "牙膏"), ("米饭", "面条"),
            ("手机", "平板"), ("篮球", "足球"), ("咖啡", "红茶"),
            ("雨伞", "遮阳伞"), ("地铁", "公交车")
        ]
        civilian_word, undercover_word = random.choice(fallback_pairs)

    state["civilian_word"] = civilian_word
    state["undercover_word"] = undercover_word
    print(f"\n✅ 词语已就绪，游戏即将开始...\n")
    return state

# ---- 节点2：分配角色 ----
def assign_roles(state: GameState) -> GameState:
    player_name = state["player_name"]
    # AI 玩家名称
    ai_agents = ["小明", "小红", "小刚", "老王"]
    all_players = [player_name] + ai_agents
    state["all_players"] = all_players

    # 随机指定卧底（可能是玩家，也可能是AI）
    undercover = random.choice(all_players)

    for p in all_players:
        if p == undercover:
            state["role_assignment"][p] = ("卧底", state["undercover_word"])
        else:
            state["role_assignment"][p] = ("平民", state["civilian_word"])

    # 记录玩家自身角色
    player_role, player_word = state["role_assignment"][player_name]
    state["player_role"] = player_role
    state["player_word"] = player_word

    # ---- 展示给玩家自己看的私密信息 ----
    print_separator("─", 55)
    print(f"🔐 【你的私密信息 - 只有你能看到】")
    print_separator("─", 55)
    slow_print(f"  你的角色：{'🕵️  卧底' if player_role == '卧底' else '👤  平民'}", 0.05)
    slow_print(f"  你的词语：【 {player_word} 】", 0.05)
    if player_role == "卧底":
        print("\n  🎯 卧底策略提示：")
        print("     • 你的词与平民词相近但不同，要模仿平民的描述风格")
        print("     • 避免直接说出词语，用模糊但合理的特征蒙混过关")
        print("     • 目标：让平民投票淘汰彼此，直到卧底人数≥平民")
    else:
        print("\n  🎯 平民策略提示：")
        print("     • 描述你的词语特征，但不要直接说出词语")
        print("     • 注意听其他人的描述，找出发言模糊/矛盾的玩家")
        print("     • 目标：找出卧底并将其淘汰")
    print_separator("─", 55)

    # 隐藏 AI 角色信息（不打印，防止泄露）
    input("\n👆 记好你的词语后，按 Enter 继续...")

    print(f"\n🎭 本局共有 {len(all_players)} 名玩家参与：")
    for p in all_players:
        tag = "（你）" if p == player_name else "（AI）"
        print(f"   • {p} {tag}")
    return state

# ---- 节点3：发言 ----
def generate_speeches(state: GameState) -> GameState:
    speeches = {}
    reasoning = {}
    current_round = state["round"]
    player_name = state["player_name"]
    active_players = [p for p in state["all_players"] if p not in state["eliminated"]]
    random.shuffle(active_players)

    # 格式化历史发言，供玩家和AI参考
    history_text = ""
    if state["history_speeches"]:
        history_text = "【历史发言记录】\n"
        for idx, round_speech in enumerate(state["history_speeches"], 1):
            history_text += f"第{idx}轮：\n"
            for p, s in round_speech.items():
                if p not in state["eliminated"]:
                    history_text += f"  {p}：{s}\n"
        history_text += "\n"

    # 兜底发言库（JSON解析失败时使用）
    fallback_speeches = [
        "这个在日常生活中挺常见的，用的地方不少。",
        "我第一个想到的就是它，很多人都用过。",
        "其实挺普通的一东西，没啥特别的。",
        "每次说到这个我都会想到一些日常场景。",
        "周围认识的人基本都用过，反馈还不错。",
        "说起来还挺有意思的，不同人有不同感受。",
        "算是比较常见的那种吧，用起来挺顺手。",
        "这个东西有时候挺重要，有时候又容易被忽略。"
    ]

    print_title(f"🗣  第 {current_round} 轮 · 发言阶段", "─", 55)
    print()

    for player in active_players:
        role, word = state["role_assignment"][player]

        if player == player_name and not state["player_eliminated"]:
            # ======== 玩家亲自发言 ========
            print(f"🎤 轮到你发言了！({player_name} · {role})")
            print(f"   你的词语是：【{word}】")
            print("   ⚠️  提示：描述词语特征，不能直接说出词语！")

            # 显示历史发言供参考
            if history_text:
                print("\n   📋 历史发言供参考：")
                for line in history_text.split("\n"):
                    if line.strip():
                        print(f"      {line}")

            while True:
                speech = input("\n✏️  请输入你的发言（5-120字）：").strip()
                if 5 <= len(speech) <= 120:
                    break
                print(f"   ⚠️  当前{len(speech)}字，请输入5-120字")

            speeches[player_name] = speech
            reasoning[player_name] = "（玩家手动输入）"
            print(f"   ✅ 已记录：『{speech}』")
        else:
            # ======== AI 发言 ========
            # 为每个AI单独构建prompt，互不干扰，避免结构重复
            if role == "平民":
                system_prompt = f"""你是「谁是卧底」游戏中的玩家「{player}」，身份是【平民】。
你的词语是「{word}」，普通平民拿到的词。

任务要求：
- 用一句话描述这个词，必须让其他玩家猜到，但不能直接说出词本身
- 字数20-50字，口语化，像正常人聊天一样说话，不要模板化
- 禁止用"生活中常见""大家应该都知道""很普及"这类万能废话
- 禁止用AI腔调，像朋友聊天时的自然表达
- 选取你感兴趣的独特角度：比如一次具体经历、某个鲜明特点、或和生活场景的关联

{history_text}
直接输出一句话即可，不要加引号，不要加名字前缀。"""
            else:
                system_prompt = f"""你是「谁是卧底」游戏中的玩家「{player}」，身份是【卧底】。
你的词语是「{word}」，它和平民的词「{state['civilian_word']}」很相似但有一点不同。

任务要求：
- 你的目标是混在平民中不被发现
- 用一句话描述，风格要和大部分人一致，不能太特别
- 字数20-50字，像正常人聊天，不能有AI腔
- 避免描述得太准确（因为你拿的不是平民词），但也不能太模糊引起怀疑
- 模仿平民的发言风格，模糊处理那1-2个差异点

{history_text}
直接输出一句话即可，不要加引号，不要加名字前缀。"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "请发言")
            ])
            chain = prompt | llm | parser

            print(f"💭 {player} 思考中...", end="", flush=True)
            try:
                speech = chain.invoke({}).strip()
                # 过滤掉名字前缀或引号污染
                speech = speech.replace(f"「{player}」", "").replace(f"{player}：", "").replace(f"{player}:", "").strip('"\'「」')
            except Exception:
                speech = random.choice(fallback_speeches)

            # 兜底：发言过短或解析失败
            if len(speech) < 8:
                speech = random.choice(fallback_speeches)

            speeches[player] = speech
            reasoning[player] = "AI生成"
            time.sleep(0.4)
            print(f"\n  💬 {player}：{speech}")

    # 存入历史
    state["history_speeches"].append(speeches.copy())
    state["speeches"] = speeches
    state["speech_reasoning"] = reasoning
    return state

# ---- 节点4：投票 ----
def vote_undercover(state: GameState) -> GameState:
    votes = {}
    reasons = {}
    current_round = state["round"]
    player_name = state["player_name"]
    active_players = [p for p in state["all_players"] if p not in state["eliminated"]]

    # 格式化发言上下文
    speech_context = f"【第{current_round}轮发言】\n"
    speech_context += "\n".join([f"{p}：{s}" for p, s in state["speeches"].items()])
    if state["history_speeches"]:
        speech_context += "\n\n【历史发言参考】\n"
        for idx, round_speeches in enumerate(state["history_speeches"][:-1], 1):
            speech_context += f"第{idx}轮：\n"
            for p, s in round_speeches.items():
                if p in active_players:
                    speech_context += f"- {p}：{s}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是「谁是卧底」游戏的理性玩家，需基于当前轮+历史发言分析并投票。
【分析规则】
1. 投票依据：
   - 对比玩家当前轮和历史发言，找出矛盾/异常的描述（卧底常出现前后矛盾）
   - 平民：重点关注发言前后不一致、描述偏离词语特征的玩家
   - 卧底：找出看起来像平民的玩家投票，避免自己被怀疑，保持投票理由连贯
2. 输出格式：必须严格按照JSON格式输出，示例：
   {{{{"vote": "agent2", "reason": "agent2本轮和上轮发言矛盾，描述不符合平民词特征"}}}}
禁止输出任何额外文字，只返回JSON字符串！
{speech_context}"""),
        ("user", """你的角色：{role}
你的词语：{word}
当前存活玩家：{players}
请选择你要投票的玩家并说明理由（理由控制在50字内）""")
    ])
    chain = prompt | llm | parser

    print_title(f"🗳  第 {current_round} 轮 · 投票阶段", "─", 55)
    print("\n  📋 本轮所有发言回顾：")
    for p, s in state["speeches"].items():
        tag = "（你）" if p == player_name else ""
        print(f"     {p}{tag}：{s}")

    # 随机决定投票顺序（玩家也在其中）
    vote_order = [p for p in active_players]
    random.shuffle(vote_order)

    for player in vote_order:
        role, word = state["role_assignment"][player]

        if player == player_name and not state["player_eliminated"]:
            # ======== 玩家亲自投票 ========
            print(f"\n🗳  轮到你投票了！（你是 {role}，词语：{word}）")
            candidates = [p for p in active_players if p != player_name]
            print("   当前存活的其他玩家：")
            for i, c in enumerate(candidates, 1):
                print(f"     {i}. {c}")

            while True:
                choice = input(f"\n✏️  请输入要投票的玩家名字或编号（1~{len(candidates)}）：").strip()
                # 支持数字编号
                if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                    vote = candidates[int(choice) - 1]
                    break
                elif choice in candidates:
                    vote = choice
                    break
                else:
                    print(f"   ⚠️  输入无效，请输入正确的玩家名字或编号")

            reason_input = input(f"✏️  请输入投票理由（可简短）：").strip()
            if not reason_input:
                reason_input = "（玩家未填写理由）"

            votes[player_name] = vote
            reasons[player_name] = reason_input
            print(f"   ✅ 你投票给了：{vote}，理由：{reason_input}")

        else:
            # ======== AI 投票 ========
            candidates_str = "、".join([p for p in active_players if p != player])
            output = chain.invoke({
                "role": role,
                "word": word,
                "speech_context": speech_context,
                "players": candidates_str
            })
            try:
                vote_data = json.loads(output.strip())
                vote = vote_data["vote"].strip()
                reason = vote_data["reason"]
            except (json.JSONDecodeError, KeyError):
                vote = random.choice([p for p in active_players if p != player])
                reason = "无有效分析，随机投票"

            # 校验投票有效性
            if vote == player or vote not in active_players:
                vote = random.choice([p for p in active_players if p != player])

            votes[player] = vote
            reasons[player] = reason
            time.sleep(0.2)
            print(f"\n  🗳  {player} 投票给：{vote}")
            print(f"      理由：{reason}")

    state["votes"] = votes
    state["vote_reasoning"] = reasons
    return state

# ---- 节点5：裁决 ----
def judge_result(state: GameState) -> GameState:
    vote_count = {}
    for v in state["votes"].values():
        vote_count[v] = vote_count.get(v, 0) + 1

    max_vote = max(vote_count.values())
    eliminated_candidate = [p for p, c in vote_count.items() if c == max_vote]

    # 打印得票统计
    print(f"\n  📊 得票统计：")
    for p, c in sorted(vote_count.items(), key=lambda x: -x[1]):
        bar = "█" * c
        tag = "（你）" if p == state["player_name"] else ""
        print(f"     {p}{tag}：{bar} {c}票")

    eliminated = random.choice(eliminated_candidate)
    state["eliminated"].append(eliminated)
    role = state["role_assignment"][eliminated][0]
    current_round = state["round"]

    print(f"\n  ❌ 第 {current_round} 轮淘汰结果：【{eliminated}】（{role}）")

    # 判断是否是玩家本人被淘汰
    if eliminated == state["player_name"]:
        state["player_eliminated"] = True
        print(f"\n  😢 你被淘汰了！你的真实词语是：【{state['player_word']}】")
        print(f"     平民词：{state['civilian_word']} | 卧底词：{state['undercover_word']}")
        if role == "卧底":
            print("  💀 你作为卧底被揪出来了，平民获胜！")
        else:
            print("  💀 你是平民但被错误投票淘汰了，继续观战吧！")
    else:
        if role == "卧底":
            print(f"  🎉 {eliminated} 果然是卧底！")
        else:
            print(f"  😬 {eliminated} 是平民，投错了……")

    remaining = [p for p in state["all_players"] if p not in state["eliminated"]]
    civ_count = sum(1 for p in remaining if state["role_assignment"][p][0] == "平民")
    uc_count = sum(1 for p in remaining if state["role_assignment"][p][0] == "卧底")

    if role == "卧底":
        state["game_status"] = "end"
        state["winner"] = "civilian"
        slow_print("\n  🎊 平民阵营胜利！", 0.04)
    elif civ_count <= uc_count:
        state["game_status"] = "end"
        state["winner"] = "undercover"
        slow_print("\n  🎊 卧底阵营胜利！", 0.04)
    else:
        state["game_status"] = "running"
        state["round"] += 1
        print(f"\n  ➡  游戏继续，进入第 {state['round']} 轮，剩余 {len(remaining)} 人")
        if remaining:
            print(f"     存活玩家：{'、'.join(remaining)}")

    return state

# ---- 节点6：游戏总结 ----
def show_final_result(state: GameState) -> GameState:
    player_name = state["player_name"]
    winner = state["winner"]
    player_role = state["player_role"]

    print("\n")
    print_title("📜  游戏结束 · 最终揭秘", "═", 55)

    # 揭示所有角色
    print("\n  🎭 角色揭秘：")
    for p in state["all_players"]:
        role, word = state["role_assignment"][p]
        tag = "（你）" if p == player_name else "    "
        elim_tag = " [已淘汰]" if p in state["eliminated"] else " [存活]"
        icon = "🕵️" if role == "卧底" else "👤"
        print(f"     {icon} {p}{tag} · {role} · 词语：{word}{elim_tag}")

    print(f"\n  📖 词语对照：")
    print(f"     平民词：【{state['civilian_word']}】  vs  卧底词：【{state['undercover_word']}】")

    # 判断玩家个人结局
    print(f"\n  🏆 最终结果：{'平民阵营胜利 🎉' if winner == 'civilian' else '卧底阵营胜利 😈'}")

    # 玩家个人战绩
    if player_role == "平民":
        if winner == "civilian":
            personal = "🥇 你作为平民，和队友一起成功揪出了卧底！"
        else:
            personal = "😔 你作为平民，不幸被卧底反杀了……"
    else:  # 卧底
        if winner == "undercover":
            personal = "🕵️ 你作为卧底，成功蒙混过关，潜伏获胜！"
        else:
            personal = "😅 你作为卧底，最终还是被平民识破了……"

    print(f"\n  🎮 你的战绩：{personal}")
    print(f"\n  📊 游戏统计：共 {state['round']} 轮 | 淘汰顺序：{'→'.join(state['eliminated'])}")
    print_separator("═", 55)
    return state

# ================== 3. 构建 LangGraph ==================
def build_game_graph():
    graph = StateGraph(GameState)
    graph.add_node("generate_words", generate_words)
    graph.add_node("assign_roles", assign_roles)
    graph.add_node("generate_speeches", generate_speeches)
    graph.add_node("vote_undercover", vote_undercover)
    graph.add_node("judge_result", judge_result)
    graph.add_node("show_final_result", show_final_result)

    graph.set_entry_point("generate_words")
    graph.add_edge("generate_words", "assign_roles")
    graph.add_edge("assign_roles", "generate_speeches")
    graph.add_edge("generate_speeches", "vote_undercover")
    graph.add_edge("vote_undercover", "judge_result")

    def route(state: GameState):
        if state["game_status"] == "running":
            return "generate_speeches"
        return "show_final_result"

    graph.add_conditional_edges("judge_result", route)
    graph.add_edge("show_final_result", END)
    return graph

# ================== 4. 入口 ==================
def main():
    print_separator("═", 55)
    print("  🎮  谁是卧底 · 人机对战版  🎮")
    print_separator("═", 55)
    print("""
  游戏规则：
  • 共 5 名玩家（1 名真实玩家 + 4 名 AI）
  • 每人持有一个词语，其中 1 人是卧底，词语与平民不同
  • 每轮所有玩家依次发言，描述自己的词语但不能直接说出
  • 发言结束后投票淘汰最可疑的玩家
  • 平民目标：找出并淘汰所有卧底
  • 卧底目标：存活到自身人数 ≥ 剩余平民人数
    """)
    print_separator("─", 55)

    # 获取玩家名字
    while True:
        player_name = input("👤 请输入你的游戏昵称：").strip()
        if player_name:
            # 防止与AI玩家重名
            reserved = ["小明", "小红", "小刚", "老王"]
            if player_name in reserved:
                print(f"  ⚠️  昵称「{player_name}」已被 AI 玩家使用，请换一个")
            else:
                break
        else:
            print("  ⚠️  昵称不能为空")

    print(f"\n  ✅ 欢迎加入，{player_name}！游戏即将开始...\n")
    input("  按 Enter 开始游戏 ▶")

    # 构建并运行游戏
    game_graph = build_game_graph()
    game = game_graph.compile()
    game.invoke(init_game_state(player_name))

    # 询问是否再玩一局
    print()
    again = input("🔄 是否再来一局？(y/n)：").strip().lower()
    if again == "y":
        main()
    else:
        print(f"\n  👋 感谢参与，{player_name}！下次再见！\n")

if __name__ == "__main__":
    main()
