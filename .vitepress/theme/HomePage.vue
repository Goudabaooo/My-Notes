<template>
  <main class="home-redesign">
    <section class="editorial-hero" aria-labelledby="home-title">
      <div class="hero-copy">
        <p class="eyebrow"><span class="spike-mark" aria-hidden="true"></span>Howard's Research Notes</p>
        <h1 id="home-title">把研究过程，整理成可以复盘的空间。</h1>
        <p class="hero-lede">
          这里存放我的日程、课程笔记、模型实验和一些随手记。它不是一次写完的作品，更像一个持续生长的研究生工作台。
        </p>
        <div class="hero-actions" aria-label="主要入口">
          <a class="home-button home-button--primary" href="/Schedule_Howard">查看近期日程</a>
          <a class="home-button home-button--secondary" href="/tips">Tips</a>
        </div>
        <dl class="hero-meta" aria-label="主页内容概览">
          <div>
            <dt>Focus</dt>
            <dd>机器学习</dd>
          </div>
          <div>
            <dt>Archive</dt>
            <dd>课程与论文</dd>
          </div>
          <div>
            <dt>Notes</dt>
            <dd>随笔与配置</dd>
          </div>
        </dl>
      </div>

      <div class="research-mockup" aria-label="研究日志预览">
        <div class="mockup-topbar">
          <span></span>
          <span></span>
          <span></span>
          <strong>research-log.md</strong>
        </div>
        <div class="mockup-body">
          <div class="mockup-column">
            <p class="mockup-label">Today</p>
            <h2>Schedule</h2>
            <ul>
              <li><span>09:30</span> Read papers</li>
              <li><span>14:00</span> Model notes</li>
              <li><span>21:00</span> Write summary</li>
            </ul>
          </div>
          <div class="code-window" aria-label="模型实验片段">
            <p><span>01</span>loss = optimize(model, data)</p>
            <p><span>02</span>calibrate(predictions)</p>
            <p><span>03</span>write_down(insight)</p>
          </div>
        </div>
      </div>
    </section>

    <section class="home-section home-section--cards" aria-labelledby="paths-title">
      <div class="section-heading">
        <p class="eyebrow"><span class="spike-mark" aria-hidden="true"></span>Paths</p>
        <h2 id="paths-title">三个最常回来的入口</h2>
      </div>

      <div class="path-grid">
        <a class="path-card" href="/Schedule_Howard">
          <span class="path-number">01</span>
          <h3>科研规划</h3>
          <p>近期安排、阶段任务和毕业课题思考，先看这里就能知道现在在推进什么。</p>
        </a>
        <a class="path-card" href="/StatisticalLearningTheory">
          <span class="path-number">02</span>
          <h3>课程与模型</h3>
          <p>统计学习理论、机器学习课程论文、神经网络优化算法等学习笔记。</p>
        </a>
        <a class="path-card" href="/tips">
          <span class="path-number">03</span>
          <h3>随笔与工具</h3>
          <p>Coding Tips、配置记录和一些生活里的灵感碎片，偏实用也偏随手。</p>
        </a>
      </div>
    </section>

    <section class="workbench-band" aria-labelledby="workbench-title">
      <div>
        <p class="eyebrow eyebrow--dark"><span class="spike-mark" aria-hidden="true"></span>Workbench</p>
        <h2 id="workbench-title">当前想法先放在这里，之后慢慢变成文章。</h2>
        <p>
          有些内容还只是问题、草稿和方向判断。它们会先留在工作台里，等到脉络清楚后再沉淀到正式笔记中。
        </p>
      </div>

      <div class="workbench-links" aria-label="当前工作台链接">
        <a href="/毕业课题%20选择">
          <span>毕业课题选择</span>
          <em>Research direction</em>
        </a>
        <a href="/神经网络优化算法模型">
          <span>神经网络优化算法</span>
          <em>Optimization notes</em>
        </a>
        <a href="/随手记">
          <span>随手记</span>
          <em>Loose thoughts</em>
        </a>
      </div>
    </section>
  </main>
</template>
