import{n as e}from"./chunk-AMVFOWMQ.CDJ5rbKU.js";import{t}from"./chunk-YJFJOXZG.RhghTvNC.js";import{C as n,G as r,S as i,k as a,l as o,o as s}from"./chunk-W7FHEGFS.DSrYjMzS.js";import{p as c}from"./render-O7CIS3YK.DF1vQoMQ.js";import{r as l}from"./mermaid-layout-elk.core.BTsxVFMD.js";var u=``,d=``,f=``,p=[],m=new Map,h=e(e=>r(e,n()),`sanitizeText`),g=e(e=>{switch(e.type){case`terminal`:return{...e,value:h(e.value)};case`nonterminal`:return{...e,name:h(e.name)};case`sequence`:return{...e,elements:e.elements.map(g)};case`choice`:return{...e,alternatives:e.alternatives.map(g)};case`optional`:return{...e,element:g(e.element)};case`repetition`:return{...e,element:g(e.element),separator:e.separator?g(e.separator):void 0};case`special`:return{...e,text:h(e.text)}}},`sanitizeAstNode`),_=e(()=>{u=``,d=``,f=``,p.length=0,m.clear(),s(),t.debug(`[Railroad] Database cleared`)},`clear`),v=e(e=>{u=h(e),t.debug(`[Railroad] Title set:`,e)},`setTitle`),y=e(()=>u,`getTitle`),b={clear:_,setTitle:v,getTitle:y,addRule:e(e=>{let n={...e,name:h(e.name),definition:g(e.definition),comment:e.comment?h(e.comment):void 0};t.debug(`[Railroad] Adding rule:`,n.name),m.has(n.name)&&t.warn(`[Railroad] Rule '${n.name}' is already defined. Overwriting.`),p.push(n),m.set(n.name,n)},`addRule`),getRules:e(()=>p,`getRules`),getRule:e(e=>m.get(e),`getRule`),setAccTitle:e(e=>{d=h(e).replace(/^\s+/g,``),t.debug(`[Railroad] Accessibility title set:`,e)},`setAccTitle`),getAccTitle:e(()=>d,`getAccTitle`),setAccDescription:e(e=>{f=h(e).replace(/\n\s+/g,`
`),t.debug(`[Railroad] Accessibility description set:`,e)},`setAccDescription`),getAccDescription:e(()=>f,`getAccDescription`),setDiagramTitle:v,getDiagramTitle:y},x={compactMode:!1,padding:10,verticalSeparation:8,horizontalSeparation:10,arcRadius:10,fontSize:14,fontFamily:`monospace`,terminalFill:`#FFFFC0`,terminalStroke:`#000000`,terminalTextColor:`#000000`,nonTerminalFill:`#FFFFFF`,nonTerminalStroke:`#000000`,nonTerminalTextColor:`#000000`,lineColor:`#000000`,strokeWidth:2,markerFill:`#000000`,commentFill:`#E8E8E8`,commentStroke:`#888888`,commentTextColor:`#666666`,specialFill:`#F0E0FF`,specialStroke:`#8800CC`,ruleNameColor:`#000066`,showMarkers:!0,markerRadius:5},S=/^#(?:[\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$|^(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\([\d\s%+,./-]+\)$|^[a-z]+$/i,C=/^[\w "',.-]+$/,w=new Set([`compactMode`,`padding`,`verticalSeparation`,`horizontalSeparation`,`arcRadius`,`fontSize`,`fontFamily`,`terminalFill`,`terminalStroke`,`terminalTextColor`,`nonTerminalFill`,`nonTerminalStroke`,`nonTerminalTextColor`,`lineColor`,`strokeWidth`,`markerFill`,`commentFill`,`commentStroke`,`commentTextColor`,`specialFill`,`specialStroke`,`ruleNameColor`,`showMarkers`,`markerRadius`]),T=e(e=>e?Object.keys(e).every(e=>e===`railroad`||w.has(e)):!1,`isRailroadStyleOptions`),E=e(e=>e?`railroad`in e&&e.railroad?e.railroad:T(e)?e:{}:{},`extractRailroadOverrides`),D=e(e=>{if(!e||T(e))return{};let{railroad:t,svgId:n,theme:r,look:i,...a}=e;return a},`extractThemeOverrides`),O=e((e,t)=>{if(typeof e!=`string`)return t;let n=e.trim();return S.test(n)?n:t},`sanitizeColorValue`),k=e((e,t)=>{if(typeof e!=`string`)return t;let n=e.trim();return C.test(n)?n:t},`sanitizeFontFamilyValue`),A=e((e,t)=>{let n=typeof e==`number`?e:typeof e==`string`?Number.parseFloat(e):NaN;return Number.isFinite(n)&&n>=0?n:t},`sanitizeNumberValue`),j=e(e=>{let t=typeof e==`number`?e:typeof e==`string`?Number.parseFloat(e):NaN;return Number.isFinite(t)&&t>0?t:void 0},`parseThemeFontSize`),M=e(e=>{let t=k(e.fontFamily,x.fontFamily),n=j(e.fontSize)??x.fontSize;return{...x,fontFamily:t,fontSize:n,terminalFill:O(e.secondBkg??e.secondaryColor,x.terminalFill),terminalStroke:O(e.secondaryBorderColor??e.lineColor,x.terminalStroke),terminalTextColor:O(e.secondaryTextColor??e.textColor,x.terminalTextColor),nonTerminalFill:O(e.mainBkg??e.background,x.nonTerminalFill),nonTerminalStroke:O(e.primaryBorderColor??e.lineColor,x.nonTerminalStroke),nonTerminalTextColor:O(e.primaryTextColor??e.textColor,x.nonTerminalTextColor),lineColor:O(e.lineColor,x.lineColor),markerFill:O(e.lineColor,x.markerFill),commentFill:O(e.labelBackground??e.tertiaryColor,x.commentFill),commentStroke:O(e.tertiaryBorderColor??e.lineColor,x.commentStroke),commentTextColor:O(e.tertiaryTextColor??e.textColor,x.commentTextColor),specialFill:O(e.tertiaryColor??e.secondaryColor,x.specialFill),specialStroke:O(e.tertiaryBorderColor??e.secondaryBorderColor,x.specialStroke),ruleNameColor:O(e.titleColor??e.textColor,x.ruleNameColor)}},`buildThemeDefaults`),N=e(e=>{let t=i(),n=M({...a(),...t.themeVariables??{},...D(e)}),r={...t.railroad??{},...E(e)};return{compactMode:r.compactMode??n.compactMode,padding:A(r.padding,n.padding),verticalSeparation:A(r.verticalSeparation,n.verticalSeparation),horizontalSeparation:A(r.horizontalSeparation,n.horizontalSeparation),arcRadius:A(r.arcRadius,n.arcRadius),fontSize:A(r.fontSize,n.fontSize),fontFamily:k(r.fontFamily,n.fontFamily),terminalFill:O(r.terminalFill,n.terminalFill),terminalStroke:O(r.terminalStroke,n.terminalStroke),terminalTextColor:O(r.terminalTextColor,n.terminalTextColor),nonTerminalFill:O(r.nonTerminalFill,n.nonTerminalFill),nonTerminalStroke:O(r.nonTerminalStroke,n.nonTerminalStroke),nonTerminalTextColor:O(r.nonTerminalTextColor,n.nonTerminalTextColor),lineColor:O(r.lineColor,n.lineColor),strokeWidth:A(r.strokeWidth,n.strokeWidth),markerFill:O(r.markerFill,n.markerFill),commentFill:O(r.commentFill,n.commentFill),commentStroke:O(r.commentStroke,n.commentStroke),commentTextColor:O(r.commentTextColor,n.commentTextColor),specialFill:O(r.specialFill,n.specialFill),specialStroke:O(r.specialStroke,n.specialStroke),ruleNameColor:O(r.ruleNameColor,n.ruleNameColor),showMarkers:r.showMarkers??n.showMarkers,markerRadius:A(r.markerRadius,n.markerRadius)}},`buildRailroadStyleOptions`),P=e(e=>{let{fontFamily:t,fontSize:n,terminalFill:r,terminalStroke:i,terminalTextColor:a,nonTerminalFill:o,nonTerminalStroke:s,nonTerminalTextColor:c,lineColor:l,strokeWidth:u,markerFill:d,commentFill:f,commentStroke:p,commentTextColor:m,specialFill:h,specialStroke:g,ruleNameColor:_}=N(e);return`
  .railroad-diagram {
    font-family: ${t};
    font-size: ${n}px;
  }

  .railroad-terminal rect {
    fill: ${r};
    stroke: ${i};
    stroke-width: ${u}px;
  }

  .railroad-terminal text {
    fill: ${a};
    font-family: ${t};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-nonterminal rect {
    fill: ${o};
    stroke: ${s};
    stroke-width: ${u}px;
  }

  .railroad-nonterminal text {
    fill: ${c};
    font-family: ${t};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-line {
    stroke: ${l};
    stroke-width: ${u}px;
    fill: none;
  }

  .railroad-start circle,
  .railroad-end circle {
    fill: ${d};
  }

  .railroad-comment ellipse {
    fill: ${f};
    stroke: ${p};
    stroke-width: ${u}px;
  }

  .railroad-comment text {
    fill: ${m};
    font-style: italic;
    font-family: ${t};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-special rect {
    fill: ${h};
    stroke: ${g};
    stroke-width: ${u}px;
    stroke-dasharray: 5,3;
  }

  .railroad-special text {
    fill: ${c};
    font-family: ${t};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-rule-name {
    font-weight: bold;
    fill: ${_};
    font-family: ${t};
    font-size: ${n}px;
  }

  .railroad-group {
    /* Grouping container, no specific styles */
  }
`},`getStyles`),F=class{static{l(this,`PathBuilder`)}constructor(){this.d=``}static{e(this,`PathBuilder`)}moveTo(e,t){return this.d+=`M ${e} ${t} `,this}lineTo(e,t){return this.d+=`L ${e} ${t} `,this}horizontalTo(e){return this.d+=`H ${e} `,this}verticalTo(e){return this.d+=`V ${e} `,this}arcTo(e,t,n,r,i,a,o){return this.d+=`A ${e} ${t} ${n} ${+!!r} ${+!!i} ${a} ${o} `,this}build(){return this.d.trim()}},I=class{static{l(this,`RailroadRenderer`)}constructor(e,t=N()){this.textCache=new Map,this.svg=e,this.config=t}static{e(this,`RailroadRenderer`)}measureText(e){if(this.textCache.has(e))return this.textCache.get(e);let t=this.svg.append(`text`).attr(`font-family`,this.config.fontFamily).attr(`font-size`,this.config.fontSize).text(e),n=t.node().getBBox(),r={width:n.width,height:n.height};return t.remove(),this.textCache.set(e,r),r}renderTerminal(e,t){let n=this.measureText(t),r=n.width+this.config.padding*2,i=n.height+this.config.padding*2,a=e.append(`g`).attr(`class`,`railroad-terminal`);return a.append(`rect`).attr(`x`,0).attr(`y`,0).attr(`width`,r).attr(`height`,i).attr(`rx`,10).attr(`ry`,10),a.append(`text`).attr(`x`,r/2).attr(`y`,i/2).text(t),{element:a.node(),dimensions:{width:r,height:i,up:i/2,down:i/2}}}renderNonTerminal(e,t){let n=this.measureText(t),r=n.width+this.config.padding*2,i=n.height+this.config.padding*2,a=e.append(`g`).attr(`class`,`railroad-nonterminal`);return a.append(`rect`).attr(`x`,0).attr(`y`,0).attr(`width`,r).attr(`height`,i),a.append(`text`).attr(`x`,r/2).attr(`y`,i/2).text(t),{element:a.node(),dimensions:{width:r,height:i,up:i/2,down:i/2}}}renderSequence(e,t){let n=t.map(t=>this.renderExpression(e,t)),r=0,i=0,a=0;for(let e of n)r+=e.dimensions.width,i=Math.max(i,e.dimensions.up),a=Math.max(a,e.dimensions.down);r+=(n.length-1)*this.config.horizontalSeparation;let o=e.append(`g`).attr(`class`,`railroad-sequence`),s=0;for(let e=0;e<n.length;e++){let t=n[e],r=i-t.dimensions.up;if(o.node().appendChild(t.element).setAttribute(`transform`,`translate(${s}, ${r})`),e<n.length-1){let e=s+t.dimensions.width,n=e+this.config.horizontalSeparation,r=i;o.append(`path`).attr(`class`,`railroad-line`).attr(`d`,new F().moveTo(e,r).lineTo(n,r).build())}s+=t.dimensions.width+this.config.horizontalSeparation}return{element:o.node(),dimensions:{width:r,height:i+a,up:i,down:a}}}renderChoice(e,t){let n=t.map(t=>this.renderExpression(e,t)),r=0,i=0;for(let e of n)r=Math.max(r,e.dimensions.width),i+=e.dimensions.height;i+=(n.length-1)*this.config.verticalSeparation;let a=this.config.arcRadius,o=a*4,s=r+o,c=e.append(`g`).attr(`class`,`railroad-choice`),l=0,u=i/2;for(let e of n){let t=l,n=t+e.dimensions.up,i=a*2+(r-e.dimensions.width)/2;c.node().appendChild(e.element).setAttribute(`transform`,`translate(${i}, ${t})`);let o=new F,d=n>u;n===u?o.moveTo(0,u).lineTo(i,n):o.moveTo(0,u).arcTo(a,a,0,!1,d,a,u+(d?a:-a)).lineTo(a,n-(d?a:-a)).arcTo(a,a,0,!1,!d,a*2,n).lineTo(i,n),c.append(`path`).attr(`class`,`railroad-line`).attr(`d`,o.build());let f=new F,p=i+e.dimensions.width,m=s-a*2;n===u?f.moveTo(p,n).lineTo(s,u):f.moveTo(p,n).lineTo(m,n).arcTo(a,a,0,!1,!d,s-a,n+(d?-a:a)).lineTo(s-a,u+(d?a:-a)).arcTo(a,a,0,!1,d,s,u),c.append(`path`).attr(`class`,`railroad-line`).attr(`d`,f.build()),l+=e.dimensions.height+this.config.verticalSeparation}return{element:c.node(),dimensions:{width:s,height:i,up:u,down:i-u}}}renderOptional(e,t){let n=this.renderExpression(e,t),r=this.config.arcRadius,i=r*2,a=n.dimensions.width+r*4,o=n.dimensions.height+i,s=e.append(`g`).attr(`class`,`railroad-optional`),c=r*2,l=i;s.node().appendChild(n.element).setAttribute(`transform`,`translate(${c}, ${l})`);let u=l+n.dimensions.up,d=new F().moveTo(0,u).lineTo(r*2,u);s.append(`path`).attr(`class`,`railroad-line`).attr(`d`,d.build());let f=new F().moveTo(c+n.dimensions.width,u).lineTo(a,u);s.append(`path`).attr(`class`,`railroad-line`).attr(`d`,f.build());let p=new F().moveTo(0,u).arcTo(r,r,0,!1,!1,r,u-r).lineTo(r,r).arcTo(r,r,0,!1,!0,r*2,0).lineTo(a-r*2,0).arcTo(r,r,0,!1,!0,a-r,r).lineTo(a-r,u-r).arcTo(r,r,0,!1,!1,a,u);return s.append(`path`).attr(`class`,`railroad-line`).attr(`d`,p.build()),{element:s.node(),dimensions:{width:a,height:o,up:u,down:o-u}}}renderRepetition(e,t,n){let r=this.renderExpression(e,t),i=this.config.arcRadius,a=i*2,o=r.dimensions.width+i*4,s=n===0,c=r.dimensions.height+a+(s?a:0),l=e.append(`g`).attr(`class`,`railroad-repetition`),u=i*2,d=s?a:0;l.node().appendChild(r.element).setAttribute(`transform`,`translate(${u}, ${d})`);let f=d+r.dimensions.up;l.append(`path`).attr(`class`,`railroad-line`).attr(`d`,new F().moveTo(0,f).lineTo(i*2,f).build()),l.append(`path`).attr(`class`,`railroad-line`).attr(`d`,new F().moveTo(u+r.dimensions.width,f).lineTo(o,f).build());let p=d+r.dimensions.height+i,m=new F().moveTo(u+r.dimensions.width,f).arcTo(i,i,0,!1,!0,u+r.dimensions.width+i,f+i).lineTo(u+r.dimensions.width+i,p).arcTo(i,i,0,!1,!0,u+r.dimensions.width,p+i).lineTo(i*2,p+i).arcTo(i,i,0,!1,!0,i,p).lineTo(i,f+i).arcTo(i,i,0,!1,!0,i*2,f);if(l.append(`path`).attr(`class`,`railroad-line`).attr(`d`,m.build()),s){let e=new F().moveTo(0,f).arcTo(i,i,0,!1,!1,i,f-i).lineTo(i,i).arcTo(i,i,0,!1,!0,i*2,0).lineTo(o-i*2,0).arcTo(i,i,0,!1,!0,o-i,i).lineTo(o-i,f-i).arcTo(i,i,0,!1,!1,o,f);l.append(`path`).attr(`class`,`railroad-line`).attr(`d`,e.build())}return{element:l.node(),dimensions:{width:o,height:c,up:f,down:c-f}}}renderSpecial(e,t){let n=this.measureText(`? `+t+` ?`),r=n.width+this.config.padding*2,i=n.height+this.config.padding*2,a=e.append(`g`).attr(`class`,`railroad-special`);return a.append(`rect`).attr(`x`,0).attr(`y`,0).attr(`width`,r).attr(`height`,i),a.append(`text`).attr(`x`,r/2).attr(`y`,i/2).text(`? `+t+` ?`),{element:a.node(),dimensions:{width:r,height:i,up:i/2,down:i/2}}}renderExpression(e,t){switch(t.type){case`terminal`:return this.renderTerminal(e,t.value);case`nonterminal`:return this.renderNonTerminal(e,t.name);case`sequence`:return this.renderSequence(e,t.elements);case`choice`:return this.renderChoice(e,t.alternatives);case`optional`:return this.renderOptional(e,t.element);case`repetition`:return this.renderRepetition(e,t.element,t.min);case`special`:return this.renderSpecial(e,t.text);default:throw Error(`Unknown node type: ${t.type}`)}}renderRule(e,t){let n=this.svg.append(`g`).attr(`class`,`railroad-rule`).attr(`transform`,`translate(0, ${t})`),r=e.name+` =`,i=this.measureText(r).width+20,a=i+20,o=n.append(`g`),s=this.renderExpression(o,e.definition),c=Math.max(20,s.dimensions.up),l=c-s.dimensions.up;return o.attr(`transform`,`translate(${a}, ${l})`),n.append(`g`).attr(`class`,`railroad-rule-name-group`).append(`text`).attr(`class`,`railroad-rule-name`).attr(`x`,0).attr(`y`,c).text(r),n.append(`g`).attr(`class`,`railroad-start`).append(`circle`).attr(`cx`,i).attr(`cy`,c).attr(`r`,this.config.markerRadius),n.append(`g`).attr(`class`,`railroad-end`).append(`circle`).attr(`cx`,a+s.dimensions.width+10).attr(`cy`,c).attr(`r`,this.config.markerRadius),n.append(`path`).attr(`class`,`railroad-line`).attr(`d`,new F().moveTo(i+this.config.markerRadius,c).lineTo(a,c).build()),n.append(`path`).attr(`class`,`railroad-line`).attr(`d`,new F().moveTo(a+s.dimensions.width,c).lineTo(a+s.dimensions.width+10-this.config.markerRadius,c).build()),{height:Math.max(40,l+s.dimensions.height+this.config.padding*2),width:a+s.dimensions.width+10+this.config.markerRadius}}renderDiagram(e){let t=this.config.padding,n=0;for(let r of e){let e=this.renderRule(r,t);t+=e.height+this.config.verticalSeparation,n=Math.max(n,e.width)}return{width:n+this.config.padding*2,height:t+this.config.padding}}},L=e((e,t,n)=>{o(e,t.height,t.width,n),e.attr(`viewBox`,`0 0 ${t.width} ${t.height}`)},`configureRailroadSvgSize`),R={draw:e((e,n,r)=>{t.debug(`[Railroad] Rendering diagram
`+e);try{let e=c(n);e.attr(`class`,`railroad-diagram`);let r=i().railroad?.useMaxWidth??!0,a=b.getRules();if(t.debug(`[Railroad] Rendering ${a.length} rules`),a.length===0){t.warn(`[Railroad] No rules to render`),L(e,{height:100,width:200},r);return}L(e,new I(e,N()).renderDiagram(a),r),t.debug(`[Railroad] Render complete`)}catch(e){throw t.error(`[Railroad] Render error:`,e),e}},`draw`)};export{P as n,R as r,b as t};