import requests
import json
import time
import csv
import os
from datetime import datetime


# ==================== 配置区 ====================

BASE_URL = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList"

# Cookie从浏览器复制，需替换为实际值
RAW_COOKIE = "GUID=09031045310045161421; nfes_isSupportWebP=1; nfes_isSupportWebP=1; UBT_VID=1783677076136.628eadcZ78KX; _RGUID=18a3b460-53a2-415d-8cbe-57abb30a4550; _ga=GA1.1.1743628088.1783682950; MKT_CKID=1783682950745.49ch8.asop; MKT_Pagesource=PC; cticket=4BCAF03655A57385FBDFAC3A190CE75FC080142D0D8B3CE435348DB6C77392D5; login_type=0; login_uid=4827BDF776256EF07D8D7A6079C011C8; DUID=u=8606C7223C482E7C1F257D292FC2F408&v=0; IsNonUser=F; AHeadUserInfo=VipGrade=0&VipGradeName=%C6%D5%CD%A8%BB%E1%D4%B1&UserName=&NoReadMessageCount=0; _udl=708D70C2B179E2F91CC5ED1C2CCE362D; Hm_lvt_a8d6737197d542432f4ff4abc6e06384=1783682950,1783686777; _ga_5DVRDQD429=GS2.1.s1783686777$o2$g0$t1783686784$j53$l0$h2009445585; _ga_B77BES1Z8Z=GS2.1.s1783686777$o2$g0$t1783686784$j53$l0$h0; _ga_9BZF483VNQ=GS2.1.s1783686777$o2$g0$t1783686784$j53$l0$h0; StartCity_Pkg=PkgStartCity=201; _bfa=1.1783677076136.628eadcZ78KX.1.1783705025576.1783705035710.5.53.290510; _jzqco=%7C%7C%7C%7C1783682950861%7C1.175301816.1783682950742.1783705025754.1783705036044.1783705025754.1783705036044.undefined.0.0.67.67"
CLEAN_COOKIE = ''.join(RAW_COOKIE.split())

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0",
    "Content-Type": "application/json; charset=utf-8",
    "Origin": "https://you.ctrip.com",
    "Referer": "https://you.ctrip.com/",
    "Accept": "*/*",
    "Cookie": CLEAN_COOKIE,
}

# 常德目标景点列表 (POI ID 和名称)
# POI ID 获取方式：携程景点详情页URL中的数字
SCENIC_SPOTS = [

    {"poi_id": 79459745, "name": "湖南常德卡乐星球欢乐世界"},
    {"poi_id": 10559092, "name": "湖南张家界国家森林公园", "max_pages": 150},
    {"poi_id": 77604, "name": "湖南长沙橘子洲景区", "max_pages": 150},
    {"poi_id": 75595, "name": "北京故宫博物院", "max_pages": 150},
    {"poi_id": 82315, "name": "重庆武隆天生三桥", "max_pages": 150},
    {"poi_id": 82204, "name": "云南石林风景区", "max_pages": 150},
    {"poi_id": 75627, "name": "上海东方明珠", "max_pages": 150},
    {"poi_id": 76744, "name": "新疆赛里木湖", "max_pages": 150},
]

# 每页评论条数
PAGE_SIZE = 20

# 请求间隔（秒）
REQUEST_INTERVAL = 1

# 输出目录
OUTPUT_DIR = "数据采集"


# ==================== 核心函数 ====================

def build_payload(poi_id, page):
    """
    构造请求负载
    参数:
        poi_id: 景点ID
        page: 页码
    返回:
        dict: 请求负载
    """
    return {
        "arg": {
            "channelType": 2,
            "collapseType": 0,
            "commentTagId": 0,
            "pageIndex": page,
            "pageSize": PAGE_SIZE,
            "poiId": poi_id,
            "sortType": 6,
            "sourceType": 1,
            "starType": 0,
            "head": {
                "cid": "09031045310045161421",
                "ctok": "",
                "cver": "1.0",
                "lang": "01",
                "sid": "8888",
                "syscode": "09",
                "auth": "",
                "extension": [],
                "xsid": ""
            }
        }
    }


def fetch_comments(poi_id, page):
    """
    抓取单页评论数据
    参数:
        poi_id: 景点ID
        page: 页码
    返回:
        dict: 接口返回的JSON数据，失败返回None
    """
    payload = build_payload(poi_id, page)
    try:
        response = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=15)
        if response.status_code != 200:
            print(f"HTTP状态码异常: {response.status_code}")
            return None
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")
        return None


def parse_comments(data):
    """
    解析评论数据，增加完整的空值防御
    参数:
        data: 接口返回的JSON数据
    返回:
        tuple: (评论列表, 总条数, 本页条数)
    """
    if not data:
        return [], 0, 0

    result = data.get('result', {})
    if result is None:
        return [], 0, 0

    items = result.get('items', [])
    if items is None:
        return [], 0, 0

    total_count = result.get('totalCount', 0)
    if total_count is None:
        total_count = 0

    comments = []
    for item in items:
        if item is None:
            continue
        user_info = item.get('userInfo', {})
        if user_info is None:
            user_info = {}
        comments.append({
            'userName': user_info.get('userNick', '') or '匿名',
            'score': item.get('score', 0) or 0,
            'content': item.get('content', '').strip() or '',
            'publishTime': item.get('publishTime', '').replace('/Date(', '').replace('+0800)/', '') or '',
            'replyCount': item.get('replyCount', 0) or 0,
            'usefulCount': item.get('usefulCount', 0) or 0,
            'touristType': item.get('touristTypeDisplay', '') or '',
            'ipLocatedName': item.get('ipLocatedName', '') or '',
        })

    return comments, total_count, len(comments)


def save_data(comments, poi_name, timestamp):
    """
    保存数据为JSON和CSV格式
    参数:
        comments: 评论列表
        poi_name: 景点名称
        timestamp: 时间戳
    """
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 清理文件名中的特殊字符
    safe_name = poi_name.replace(" ", "_").replace("/", "_")

    # JSON文件
    json_file = os.path.join(OUTPUT_DIR, f"{safe_name}_comments_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    print(f"JSON已保存: {json_file}")

    # CSV文件
    if comments:
        csv_file = os.path.join(OUTPUT_DIR, f"{safe_name}_comments_{timestamp}.csv")
        fieldnames = ['userName', 'score', 'content', 'publishTime', 'replyCount',
                      'usefulCount', 'touristType', 'ipLocatedName']
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comments)
        print(f"CSV已保存: {csv_file}")


def crawl_scenic_spot(poi_id, poi_name, max_pages):
    """
    爬取单个景点的全部评论
    参数:
        poi_id: 景点ID
        poi_name: 景点名称
        max_pages: 最大爬取页数
    返回:
        list: 该景点全部评论列表
    """
    print(f"\n开始采集: {poi_name} (POI ID: {poi_id})")
    print("-" * 40)

    all_comments = []

    # 获取第一页，同时获得总条数
    print("获取总评论数...", end=" ")
    first_data = fetch_comments(poi_id, 1)
    if not first_data:
        print("失败，该景点可能无评论或接口异常")
        return []

    first_comments, total_count, _ = parse_comments(first_data)
    all_comments.extend(first_comments)
    print(f"第一页 {len(first_comments)} 条，总评论数: {total_count}")

    if total_count == 0 or len(first_comments) == 0:
        print("该景点暂无评论")
        return []

    # 计算总页数（取实际页数和配置上限的较小值）
    actual_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = min(actual_pages, max_pages) if max_pages else actual_pages
    print(f"实际总页数: {actual_pages}，本次采集: {total_pages} 页")

    # 从第二页开始循环抓取
    for page in range(2, total_pages + 1):
        print(f"正在抓取第 {page}/{total_pages} 页...", end=" ")
        data = fetch_comments(poi_id, page)
        if not data:
            print("请求失败，终止")
            break

        comments, _, item_count = parse_comments(data)
        all_comments.extend(comments)
        print(f"获取 {len(comments)} 条 (累计 {len(all_comments)}/{total_count} 条)")

        if len(comments) == 0:
            print("本页无评论，提前终止")
            break

        time.sleep(REQUEST_INTERVAL)

    print(f"{poi_name} 采集完成，共 {len(all_comments)} 条评论")
    return all_comments


def main():
    """
    主函数：执行爬虫任务
    """
    print("=" * 50)
    print("常德景点评论数据采集程序")
    print(f"目标景点数量: {len(SCENIC_SPOTS)}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for spot in SCENIC_SPOTS:
        poi_id = spot.get("poi_id")
        poi_name = spot.get("name")
        max_pages = spot.get("max_pages", 0)  # 0表示不设上限，爬取全部

        comments = crawl_scenic_spot(poi_id, poi_name, max_pages)

        if comments:
            save_data(comments, poi_name, timestamp)
        else:
            print(f"{poi_name} 未获取到有效数据")

        # 景点之间增加间隔，防止被封
        time.sleep(3)

    print("\n" + "=" * 50)
    print("全部采集任务完成")
    print("=" * 50)


if __name__ == "__main__":
    main()