<template>
  <div class="shell">
    <nav class="nav">
      <h1>EmbodiedBein</h1>
      <router-link to="/">总览</router-link>
      <router-link to="/contracts">契约</router-link>
      <router-link to="/data">数据</router-link>
      <router-link to="/models">模型</router-link>
      <router-link to="/train/experiments">实验</router-link>
      <router-link to="/train/submit">提交任务</router-link>
      <router-link to="/train/queue">队列</router-link>
      <router-link to="/eval/suites">评估</router-link>
      <router-link to="/eval/leaderboard">排行榜</router-link>
      <router-link to="/deploy/bundles">部署</router-link>
      <router-link to="/agent">Agent</router-link>
      <router-link to="/showcase/bhl-progress-demo">Showcase</router-link>
    </nav>
    <main class="main"><router-view /></main>
  </div>
</template>
