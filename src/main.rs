//! Mini-RecSys - 混合 Rust/C++ 推荐系统 Demo

mod ffi;
mod model;

use model::init_data;
use std::sync::Arc;

fn main() {
    println!("=== Mini-RecSys: Rust/C++ FFI Demo ===\n");

    // 初始化应用状态（Arc<AppState>）
    // Arc 提供线程安全的只读共享：
    // - 数据在初始化后不再修改，所以只需要读取权限
    // - Arc 通过原子计数实现共享所有权，多个线程可同时持有引用
    // - 不需要 Mutex：Mutex 用于保护可变数据的互斥访问
    //   而这里的数据是只读的，多个线程并发读取完全安全
    let state = init_data();
    
    println!("📊 数据初始化完成:");
    println!("   用户数: {}", state.users.len());
    println!("   物品数: {}", state.items.len());
    println!("   向量维度: {}\n", state.users[0].embedding.len());

    // 测试 1: FFI 加法
    println!("📝 测试 1: C++ 加法函数");
    let sum = ffi::add(42, 58);
    println!("   42 + 58 = {}", sum);
    println!("   ✅ FFI 调用成功!\n");

    // 测试 2: 召回测试
    println!("📝 测试 2: 推荐召回");
    let user = &state.users[0];
    let results = ffi::recommend_recall(&user.embedding, &state.items, 5);
    
    println!("   用户 {} 的 Top 5 推荐:", user.id);
    for (item_id, score) in &results {
        println!("   - Item {}: score = {:.4}", item_id, score);
    }
    println!("   ✅ 召回成功!\n");

    // 演示 Arc 的多线程共享能力
    let state_clone = Arc::clone(&state);
    println!("📝 Arc 引用计数: {}", Arc::strong_count(&state));
    drop(state_clone);
    println!("   drop 后计数: {}\n", Arc::strong_count(&state));

    println!("=== 阶段 3 完成 ===");
}
