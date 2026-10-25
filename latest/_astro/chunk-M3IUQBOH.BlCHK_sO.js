import{X as e}from"./src.B2_BndWA.js";import{n as t}from"./chunk-AMVFOWMQ.C_4fk-Dj.js";import{t as n}from"./chunk-YJFJOXZG.wtGER9Hz.js";import{C as r,E as i,J as a,L as o,Q as s,Y as c,b as l,c as u,o as d,x as f}from"./chunk-W7FHEGFS.CvoMBbdl.js";import{c as p,v as m}from"./chunk-S2UQUSRU.As3sCcV3.js";import{t as h}from"./chunk-LV3HUYFY.BMqR2kbM.js";import{t as g}from"./chunk-NDIOYS4C.C1ED6Mj7.js";import{t as _}from"./chunk-ISA7TJRJ.D5kdSIFH.js";import{f as v}from"./render-O7CIS3YK.DfHZqklC.js";import{r as y}from"./mermaid-layout-elk.core.B6NU23ls.js";var b=(function(){var e=t(function(e,t,n,r){for(n||={},r=e.length;r--;n[e[r]]=t);return n},`o`),n=[1,2],r=[1,3],i=[1,4],a=[2,4],o=[1,9],s=[1,11],c=[1,16],l=[1,17],u=[1,18],d=[1,19],f=[1,33],p=[1,20],m=[1,21],h=[1,22],g=[1,23],_=[1,24],v=[1,26],b=[1,27],x=[1,28],S=[1,29],C=[1,30],w=[1,31],T=[1,32],E=[1,35],D=[1,36],O=[1,37],k=[1,38],A=[1,34],j=[1,4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],M=[1,4,5,14,15,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,39,40,41,45,48,51,52,53,54,57],N=[4,5,16,17,19,21,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],P={trace:t(y(function(){},`trace`),`trace`),yy:{},symbols_:{error:2,start:3,SPACE:4,NL:5,SD:6,document:7,line:8,statement:9,classDefStatement:10,styleStatement:11,cssClassStatement:12,idStatement:13,DESCR:14,"-->":15,HIDE_EMPTY:16,scale:17,WIDTH:18,COMPOSIT_STATE:19,STRUCT_START:20,STRUCT_STOP:21,STATE_DESCR:22,AS:23,ID:24,FORK:25,JOIN:26,CHOICE:27,CONCURRENT:28,note:29,notePosition:30,NOTE_TEXT:31,direction:32,acc_title:33,acc_title_value:34,acc_descr:35,acc_descr_value:36,acc_descr_multiline_value:37,CLICK:38,STRING:39,HREF:40,classDef:41,CLASSDEF_ID:42,CLASSDEF_STYLEOPTS:43,DEFAULT:44,style:45,STYLE_IDS:46,STYLEDEF_STYLEOPTS:47,class:48,CLASSENTITY_IDS:49,STYLECLASS:50,direction_tb:51,direction_bt:52,direction_rl:53,direction_lr:54,eol:55,";":56,EDGE_STATE:57,STYLE_SEPARATOR:58,left_of:59,right_of:60,$accept:0,$end:1},terminals_:{2:`error`,4:`SPACE`,5:`NL`,6:`SD`,14:`DESCR`,15:`-->`,16:`HIDE_EMPTY`,17:`scale`,18:`WIDTH`,19:`COMPOSIT_STATE`,20:`STRUCT_START`,21:`STRUCT_STOP`,22:`STATE_DESCR`,23:`AS`,24:`ID`,25:`FORK`,26:`JOIN`,27:`CHOICE`,28:`CONCURRENT`,29:`note`,31:`NOTE_TEXT`,33:`acc_title`,34:`acc_title_value`,35:`acc_descr`,36:`acc_descr_value`,37:`acc_descr_multiline_value`,38:`CLICK`,39:`STRING`,40:`HREF`,41:`classDef`,42:`CLASSDEF_ID`,43:`CLASSDEF_STYLEOPTS`,44:`DEFAULT`,45:`style`,46:`STYLE_IDS`,47:`STYLEDEF_STYLEOPTS`,48:`class`,49:`CLASSENTITY_IDS`,50:`STYLECLASS`,51:`direction_tb`,52:`direction_bt`,53:`direction_rl`,54:`direction_lr`,56:`;`,57:`EDGE_STATE`,58:`STYLE_SEPARATOR`,59:`left_of`,60:`right_of`},productions_:[0,[3,2],[3,2],[3,2],[7,0],[7,2],[8,2],[8,1],[8,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,3],[9,4],[9,1],[9,2],[9,1],[9,4],[9,3],[9,6],[9,1],[9,1],[9,1],[9,1],[9,4],[9,4],[9,1],[9,2],[9,2],[9,1],[9,5],[9,5],[10,3],[10,3],[11,3],[12,3],[32,1],[32,1],[32,1],[32,1],[55,1],[55,1],[13,1],[13,1],[13,3],[13,3],[30,1],[30,1]],performAction:t(y(function(e,t,n,r,i,a,o){var s=a.length-1;switch(i){case 3:return r.setRootDoc(a[s]),a[s];case 4:this.$=[];break;case 5:a[s]!=`nl`&&(a[s-1].push(a[s]),this.$=a[s-1]);break;case 6:case 7:this.$=a[s];break;case 8:this.$=`nl`;break;case 12:this.$=a[s];break;case 13:let e=a[s-1];e.description=r.trimColon(a[s]),this.$=e;break;case 14:this.$={stmt:`relation`,state1:a[s-2],state2:a[s]};break;case 15:let t=r.trimColon(a[s]);this.$={stmt:`relation`,state1:a[s-3],state2:a[s-1],description:t};break;case 19:this.$={stmt:`state`,id:a[s-3],type:`default`,description:``,doc:a[s-1]};break;case 20:var c=a[s],l=a[s-2].trim();if(a[s].match(`:`)){var u=a[s].split(`:`);c=u[0],l=[l,u[1]]}this.$={stmt:`state`,id:c,type:`default`,description:l};break;case 21:this.$={stmt:`state`,id:a[s-3],type:`default`,description:a[s-5],doc:a[s-1]};break;case 22:this.$={stmt:`state`,id:a[s],type:`fork`};break;case 23:this.$={stmt:`state`,id:a[s],type:`join`};break;case 24:this.$={stmt:`state`,id:a[s],type:`choice`};break;case 25:this.$={stmt:`state`,id:r.getDividerId(),type:`divider`};break;case 26:this.$={stmt:`state`,id:a[s-1].trim(),note:{position:a[s-2].trim(),text:a[s].trim()}};break;case 29:this.$=a[s].trim(),r.setAccTitle(this.$);break;case 30:case 31:this.$=a[s].trim(),r.setAccDescription(this.$);break;case 32:this.$={stmt:`click`,id:a[s-3],url:a[s-2],tooltip:a[s-1]};break;case 33:this.$={stmt:`click`,id:a[s-3],url:a[s-1],tooltip:``};break;case 34:case 35:this.$={stmt:`classDef`,id:a[s-1].trim(),classes:a[s].trim()};break;case 36:this.$={stmt:`style`,id:a[s-1].trim(),styleClass:a[s].trim()};break;case 37:this.$={stmt:`applyClass`,id:a[s-1].trim(),styleClass:a[s].trim()};break;case 38:r.setDirection(`TB`),this.$={stmt:`dir`,value:`TB`};break;case 39:r.setDirection(`BT`),this.$={stmt:`dir`,value:`BT`};break;case 40:r.setDirection(`RL`),this.$={stmt:`dir`,value:`RL`};break;case 41:r.setDirection(`LR`),this.$={stmt:`dir`,value:`LR`};break;case 44:case 45:this.$={stmt:`state`,id:a[s].trim(),type:`default`,description:``};break;case 46:this.$={stmt:`state`,id:a[s-2].trim(),classes:[a[s].trim()],type:`default`,description:``};break;case 47:this.$={stmt:`state`,id:a[s-2].trim(),classes:[a[s].trim()],type:`default`,description:``}}},`anonymous`),`anonymous`),table:[{3:1,4:n,5:r,6:i},{1:[3]},{3:5,4:n,5:r,6:i},{3:6,4:n,5:r,6:i},e([1,4,5,16,17,19,22,24,25,26,27,28,29,33,35,37,38,41,45,48,51,52,53,54,57],a,{7:7}),{1:[2,1]},{1:[2,2]},{1:[2,3],4:o,5:s,8:8,9:10,10:12,11:13,12:14,13:15,16:c,17:l,19:u,22:d,24:f,25:p,26:m,27:h,28:g,29:_,32:25,33:v,35:b,37:x,38:S,41:C,45:w,48:T,51:E,52:D,53:O,54:k,57:A},e(j,[2,5]),{9:39,10:12,11:13,12:14,13:15,16:c,17:l,19:u,22:d,24:f,25:p,26:m,27:h,28:g,29:_,32:25,33:v,35:b,37:x,38:S,41:C,45:w,48:T,51:E,52:D,53:O,54:k,57:A},e(j,[2,7]),e(j,[2,8]),e(j,[2,9]),e(j,[2,10]),e(j,[2,11]),e(j,[2,12],{14:[1,40],15:[1,41]}),e(j,[2,16]),{18:[1,42]},e(j,[2,18],{20:[1,43]}),{23:[1,44]},e(j,[2,22]),e(j,[2,23]),e(j,[2,24]),e(j,[2,25]),{30:45,31:[1,46],59:[1,47],60:[1,48]},e(j,[2,28]),{34:[1,49]},{36:[1,50]},e(j,[2,31]),{13:51,24:f,57:A},{42:[1,52],44:[1,53]},{46:[1,54]},{49:[1,55]},e(M,[2,44],{58:[1,56]}),e(M,[2,45],{58:[1,57]}),e(j,[2,38]),e(j,[2,39]),e(j,[2,40]),e(j,[2,41]),e(j,[2,6]),e(j,[2,13]),{13:58,24:f,57:A},e(j,[2,17]),e(N,a,{7:59}),{24:[1,60]},{24:[1,61]},{23:[1,62]},{24:[2,48]},{24:[2,49]},e(j,[2,29]),e(j,[2,30]),{39:[1,63],40:[1,64]},{43:[1,65]},{43:[1,66]},{47:[1,67]},{50:[1,68]},{24:[1,69]},{24:[1,70]},e(j,[2,14],{14:[1,71]}),{4:o,5:s,8:8,9:10,10:12,11:13,12:14,13:15,16:c,17:l,19:u,21:[1,72],22:d,24:f,25:p,26:m,27:h,28:g,29:_,32:25,33:v,35:b,37:x,38:S,41:C,45:w,48:T,51:E,52:D,53:O,54:k,57:A},e(j,[2,20],{20:[1,73]}),{31:[1,74]},{24:[1,75]},{39:[1,76]},{39:[1,77]},e(j,[2,34]),e(j,[2,35]),e(j,[2,36]),e(j,[2,37]),e(M,[2,46]),e(M,[2,47]),e(j,[2,15]),e(j,[2,19]),e(N,a,{7:78}),e(j,[2,26]),e(j,[2,27]),{5:[1,79]},{5:[1,80]},{4:o,5:s,8:8,9:10,10:12,11:13,12:14,13:15,16:c,17:l,19:u,21:[1,81],22:d,24:f,25:p,26:m,27:h,28:g,29:_,32:25,33:v,35:b,37:x,38:S,41:C,45:w,48:T,51:E,52:D,53:O,54:k,57:A},e(j,[2,32]),e(j,[2,33]),e(j,[2,21])],defaultActions:{5:[2,1],6:[2,2],47:[2,48],48:[2,49]},parseError:t(y(function(e,t){if(t.recoverable)this.trace(e);else{var n=Error(e);throw n.hash=t,n}},`parseError`),`parseError`),parse:t(y(function(e){var n=this,r=[0],i=[],a=[null],o=[],s=this.table,c=``,l=0,u=0,d=0,f=2,p=1,m=o.slice.call(arguments,1),h=Object.create(this.lexer),g={yy:{}};for(var _ in this.yy)Object.prototype.hasOwnProperty.call(this.yy,_)&&(g.yy[_]=this.yy[_]);h.setInput(e,g.yy),g.yy.lexer=h,g.yy.parser=this,h.yylloc===void 0&&(h.yylloc={});var v=h.yylloc;o.push(v);var b=h.options&&h.options.ranges;this.parseError=typeof g.yy.parseError==`function`?g.yy.parseError:Object.getPrototypeOf(this).parseError;function x(e){r.length-=2*e,a.length-=e,o.length-=e}y(x,`popStack`),t(x,`popStack`);function S(){var e=i.pop()||h.lex()||p;return typeof e!=`number`&&(e instanceof Array&&(i=e,e=i.pop()),e=n.symbols_[e]||e),e}y(S,`lex`),t(S,`lex`);for(var C,w,T,E,D,O={},k,A,j,M;;){if(T=r[r.length-1],this.defaultActions[T]?E=this.defaultActions[T]:(C??=S(),E=s[T]&&s[T][C]),E===void 0||!E.length||!E[0]){var N=``;for(k in M=[],s[T])this.terminals_[k]&&k>f&&M.push(`'`+this.terminals_[k]+`'`);N=h.showPosition?`Parse error on line `+(l+1)+`:
`+h.showPosition()+`
Expecting `+M.join(`, `)+`, got '`+(this.terminals_[C]||C)+`'`:`Parse error on line `+(l+1)+`: Unexpected `+(C==p?`end of input`:`'`+(this.terminals_[C]||C)+`'`),this.parseError(N,{text:h.match,token:this.terminals_[C]||C,line:h.yylineno,loc:v,expected:M})}if(E[0]instanceof Array&&E.length>1)throw Error(`Parse Error: multiple actions possible at state: `+T+`, token: `+C);switch(E[0]){case 1:r.push(C),a.push(h.yytext),o.push(h.yylloc),r.push(E[1]),C=null,w?(C=w,w=null):(u=h.yyleng,c=h.yytext,l=h.yylineno,v=h.yylloc,d>0&&d--);break;case 2:if(A=this.productions_[E[1]][1],O.$=a[a.length-A],O._$={first_line:o[o.length-(A||1)].first_line,last_line:o[o.length-1].last_line,first_column:o[o.length-(A||1)].first_column,last_column:o[o.length-1].last_column},b&&(O._$.range=[o[o.length-(A||1)].range[0],o[o.length-1].range[1]]),D=this.performAction.apply(O,[c,u,l,g.yy,E[1],a,o].concat(m)),D!==void 0)return D;A&&(r=r.slice(0,-1*A*2),a=a.slice(0,-1*A),o=o.slice(0,-1*A)),r.push(this.productions_[E[1]][0]),a.push(O.$),o.push(O._$),j=s[r[r.length-2]][r[r.length-1]],r.push(j);break;case 3:return!0}}return!0},`parse`),`parse`)};P.lexer=(function(){return{EOF:1,parseError:t(y(function(e,t){if(this.yy.parser)this.yy.parser.parseError(e,t);else throw Error(e)},`parseError`),`parseError`),setInput:t(function(e,t){return this.yy=t||this.yy||{},this._input=e,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match=``,this.conditionStack=[`INITIAL`],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},`setInput`),input:t(function(){var e=this._input[0];return this.yytext+=e,this.yyleng++,this.offset++,this.match+=e,this.matched+=e,e.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),e},`input`),unput:t(function(e){var t=e.length,n=e.split(/(?:\r\n?|\n)/g);this._input=e+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-t),this.offset-=t;var r=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),n.length-1&&(this.yylineno-=n.length-1);var i=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:n?(n.length===r.length?this.yylloc.first_column:0)+r[r.length-n.length].length-n[0].length:this.yylloc.first_column-t},this.options.ranges&&(this.yylloc.range=[i[0],i[0]+this.yyleng-t]),this.yyleng=this.yytext.length,this},`unput`),more:t(function(){return this._more=!0,this},`more`),reject:t(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError(`Lexical error on line `+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:``,token:null,line:this.yylineno});return this},`reject`),less:t(function(e){this.unput(this.match.slice(e))},`less`),pastInput:t(function(){var e=this.matched.substr(0,this.matched.length-this.match.length);return(e.length>20?`...`:``)+e.substr(-20).replace(/\n/g,``)},`pastInput`),upcomingInput:t(function(){var e=this.match;return e.length<20&&(e+=this._input.substr(0,20-e.length)),(e.substr(0,20)+(e.length>20?`...`:``)).replace(/\n/g,``)},`upcomingInput`),showPosition:t(function(){var e=this.pastInput(),t=Array(e.length+1).join(`-`);return e+this.upcomingInput()+`
`+t+`^`},`showPosition`),test_match:t(function(e,t){var n,r,i;if(this.options.backtrack_lexer&&(i={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(i.yylloc.range=this.yylloc.range.slice(0))),r=e[0].match(/(?:\r\n?|\n).*/g),r&&(this.yylineno+=r.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:r?r[r.length-1].length-r[r.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+e[0].length},this.yytext+=e[0],this.match+=e[0],this.matches=e,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(e[0].length),this.matched+=e[0],n=this.performAction.call(this,this.yy,this,t,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),n)return n;if(this._backtrack){for(var a in i)this[a]=i[a];return!1}return!1},`test_match`),next:t(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var e,t,n,r;this._more||(this.yytext=``,this.match=``);for(var i=this._currentRules(),a=0;a<i.length;a++)if(n=this._input.match(this.rules[i[a]]),n&&(!t||n[0].length>t[0].length)){if(t=n,r=a,this.options.backtrack_lexer){if(e=this.test_match(n,i[a]),e!==!1)return e;if(this._backtrack){t=!1;continue}return!1}if(!this.options.flex)break}return t?(e=this.test_match(t,i[r]),e!==!1&&e):this._input===``?this.EOF:this.parseError(`Lexical error on line `+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:``,token:null,line:this.yylineno})},`next`),lex:t(y(function(){return this.next()||this.lex()},`lex`),`lex`),begin:t(y(function(e){this.conditionStack.push(e)},`begin`),`begin`),popState:t(y(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},`popState`),`popState`),_currentRules:t(y(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},`_currentRules`),`_currentRules`),topState:t(y(function(e){return e=this.conditionStack.length-1-Math.abs(e||0),e>=0?this.conditionStack[e]:`INITIAL`},`topState`),`topState`),pushState:t(y(function(e){this.begin(e)},`pushState`),`pushState`),stateStackSize:t(y(function(){return this.conditionStack.length},`stateStackSize`),`stateStackSize`),options:{"case-insensitive":!0},performAction:t(y(function(e,n,r,i){function a(){let t=n.yytext.indexOf(`%%`);if(t===0)return!1;if(t>0){let r=n.yytext.slice(0,t),i=n.yytext.slice(t);i&&e.lexer.unput(i),n.yytext=r}return!0}switch(y(a,`processId`),t(a,`processId`),r){case 0:return 38;case 1:return 40;case 2:return 39;case 3:return 44;case 4:return 51;case 5:return 52;case 6:return 53;case 7:return 54;case 8:return 5;case 9:break;case 10:break;case 11:break;case 12:break;case 13:return this.pushState(`SCALE`),17;case 14:return 18;case 15:this.popState();break;case 16:return this.begin(`acc_title`),33;case 17:return this.popState(),`acc_title_value`;case 18:return this.begin(`acc_descr`),35;case 19:return this.popState(),`acc_descr_value`;case 20:this.begin(`acc_descr_multiline`);break;case 21:this.popState();break;case 22:return`acc_descr_multiline_value`;case 23:return this.pushState(`CLASSDEF`),41;case 24:return this.popState(),this.pushState(`CLASSDEFID`),`DEFAULT_CLASSDEF_ID`;case 25:return this.popState(),this.pushState(`CLASSDEFID`),42;case 26:return this.popState(),43;case 27:return this.pushState(`CLASS`),48;case 28:return this.popState(),this.pushState(`CLASS_STYLE`),49;case 29:return this.popState(),50;case 30:return this.pushState(`STYLE`),45;case 31:return this.popState(),this.pushState(`STYLEDEF_STYLES`),46;case 32:return this.popState(),47;case 33:return this.pushState(`SCALE`),17;case 34:return 18;case 35:this.popState();break;case 36:this.pushState(`STATE`);break;case 37:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),25;case 38:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),26;case 39:return this.popState(),n.yytext=n.yytext.slice(0,-10).trim(),27;case 40:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),25;case 41:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),26;case 42:return this.popState(),n.yytext=n.yytext.slice(0,-10).trim(),27;case 43:return 51;case 44:return 52;case 45:return 53;case 46:return 54;case 47:this.pushState(`STATE_STRING`);break;case 48:return this.pushState(`STATE_ID`),`AS`;case 49:return a()?(this.popState(),`ID`):void 0;case 50:this.popState();break;case 51:return`STATE_DESCR`;case 52:throw Error(`Error: State name must be a single word. Found: "`+n.yytext.trim()+`"`);case 53:return 19;case 54:this.popState();break;case 55:return this.popState(),this.pushState(`struct`),20;case 56:return this.popState(),21;case 57:break;case 58:return this.begin(`NOTE`),29;case 59:return this.popState(),this.pushState(`NOTE_ID`),59;case 60:return this.popState(),this.pushState(`NOTE_ID`),60;case 61:this.popState(),this.pushState(`FLOATING_NOTE`);break;case 62:return this.popState(),this.pushState(`FLOATING_NOTE_ID`),`AS`;case 63:break;case 64:return`NOTE_TEXT`;case 65:return a()?(this.popState(),`ID`):void 0;case 66:return a()?(this.popState(),this.pushState(`NOTE_TEXT`),24):void 0;case 67:return this.popState(),n.yytext=n.yytext.substr(2).trim(),31;case 68:return this.popState(),n.yytext=n.yytext.slice(0,-8).trim(),31;case 69:return 6;case 70:return 6;case 71:return 16;case 72:return 57;case 73:return a()?24:void 0;case 74:return n.yytext=n.yytext.trim(),14;case 75:return 15;case 76:return 28;case 77:return 58;case 78:return 5;case 79:return`INVALID`}},`anonymous`),`anonymous`),rules:[/^(?:click\b)/i,/^(?:href\b)/i,/^(?:"[^"]*")/i,/^(?:default\b)/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:[\n]+)/i,/^(?:[\s]+)/i,/^(?:((?!\n)\s)+)/i,/^(?:#[^\n]*)/i,/^(?:%%(?!\{)[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:classDef\s+)/i,/^(?:DEFAULT\s+)/i,/^(?:\w+\s+)/i,/^(?:[^\n]*)/i,/^(?:class\s+)/i,/^(?:(\w+)+((,\s*\w+)*))/i,/^(?:[^\n]*)/i,/^(?:style\s+)/i,/^(?:[\w,]+\s+)/i,/^(?:[^\n]*)/i,/^(?:scale\s+)/i,/^(?:\d+)/i,/^(?:\s+width\b)/i,/^(?:state\s+)/i,/^(?:.*<<fork>>)/i,/^(?:.*<<join>>)/i,/^(?:.*<<choice>>)/i,/^(?:.*\[\[fork\]\])/i,/^(?:.*\[\[join\]\])/i,/^(?:.*\[\[choice\]\])/i,/^(?:.*direction\s+TB[^\n]*)/i,/^(?:.*direction\s+BT[^\n]*)/i,/^(?:.*direction\s+RL[^\n]*)/i,/^(?:.*direction\s+LR[^\n]*)/i,/^(?:["])/i,/^(?:\s*as\s+)/i,/^(?:[^\n\{]*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:\w+\s+\w+.*?\{)/i,/^(?:[^\n\s\{]+)/i,/^(?:\n)/i,/^(?:\{)/i,/^(?:\})/i,/^(?:[\n])/i,/^(?:note\s+)/i,/^(?:left of\b)/i,/^(?:right of\b)/i,/^(?:")/i,/^(?:\s*as\s*)/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:[^\n]*)/i,/^(?:\s*[^:\n\s\-]+)/i,/^(?:\s*:[^:\n;]+)/i,/^(?:[\s\S]*?\n\s*end note\b)/i,/^(?:stateDiagram\s+)/i,/^(?:stateDiagram-v2\s+)/i,/^(?:hide empty description\b)/i,/^(?:\[\*\])/i,/^(?:[^:\n\s\-\{]+)/i,/^(?:\s*:(?:[^:\n;]|:[^:\n;])+)/i,/^(?:-->)/i,/^(?:--)/i,/^(?::::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{LINE:{rules:[10,11,12],inclusive:!1},struct:{rules:[10,11,12,23,27,30,36,43,44,45,46,56,57,58,72,73,74,75,76,77],inclusive:!1},FLOATING_NOTE_ID:{rules:[65],inclusive:!1},FLOATING_NOTE:{rules:[62,63,64],inclusive:!1},NOTE_TEXT:{rules:[67,68],inclusive:!1},NOTE_ID:{rules:[66],inclusive:!1},NOTE:{rules:[59,60,61],inclusive:!1},STYLEDEF_STYLEOPTS:{rules:[],inclusive:!1},STYLEDEF_STYLES:{rules:[32],inclusive:!1},STYLE_IDS:{rules:[],inclusive:!1},STYLE:{rules:[31],inclusive:!1},CLASS_STYLE:{rules:[29],inclusive:!1},CLASS:{rules:[28],inclusive:!1},CLASSDEFID:{rules:[26],inclusive:!1},CLASSDEF:{rules:[24,25],inclusive:!1},acc_descr_multiline:{rules:[21,22],inclusive:!1},acc_descr:{rules:[19],inclusive:!1},acc_title:{rules:[17],inclusive:!1},SCALE:{rules:[14,15,34,35],inclusive:!1},ALIAS:{rules:[],inclusive:!1},STATE_ID:{rules:[49],inclusive:!1},STATE_STRING:{rules:[50,51],inclusive:!1},FORK_STATE:{rules:[],inclusive:!1},STATE:{rules:[10,11,12,37,38,39,40,41,42,47,48,52,53,54,55],inclusive:!1},ID:{rules:[10,11,12],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,8,9,11,12,13,16,18,20,23,27,30,33,36,55,58,69,70,71,72,73,74,75,77,78,79],inclusive:!0}}}})();function F(){this.yy={}}return y(F,`Parser`),t(F,`Parser`),F.prototype=P,P.Parser=F,new F})();b.parser=b;var x=b,S=`TB`,C=`TB`,w=`dir`,T=`state`,E=`root`,D=`relation`,O=`classDef`,k=`style`,A=`applyClass`,j=`default`,M=`divider`,N=`fill:none`,P=`fill: #333`,F=`c`,I=`markdown`,ee=`normal`,L=`rect`,R=`rectWithTitle`,te=`stateStart`,ne=`stateEnd`,re=`divider`,ie=`roundedWithTitle`,ae=`note`,oe=`noteGroup`,z=`statediagram`,se=`${z}-state`,ce=`transition`,le=`note`,ue=`${ce} note-edge`,de=`${z}-${le}`,fe=`${z}-cluster`,pe=`${z}-cluster-alt`,B=`parent`,V=`note`,me=`state`,H=`----`,he=`${H}${V}`,U=`${H}${B}`,W=t((e,t=C)=>{if(!e.doc)return t;let n=t;for(let t of e.doc)t.stmt===`dir`&&(n=t.value);return n},`getDir`),ge={getClasses:t(function(e,t){return t.db.getClasses()},`getClasses`),draw:t(async function(e,t,i,a){n.info(`REF0:`),n.info(`Drawing state diagram (v2)`,t);let{securityLevel:o,state:s,layout:c}=r();a.db.extract(a.db.getRootDocV2());let l=a.db.getData(),u=g(t,o);l.type=a.type,l.layoutAlgorithm=c,l.nodeSpacing=s?.nodeSpacing||50,l.rankSpacing=s?.rankSpacing||50,l.markers=r().look===`neo`?[`barbNeo`]:[`barb`],l.diagramId=t,await v(l,u);try{(typeof a.db.getLinks==`function`?a.db.getLinks():new Map).forEach((e,t)=>{let r=typeof t==`string`?t:typeof t?.id==`string`?t.id:``,i=l.nodes.find(e=>e.id===r);if(!r){n.warn(`⚠️ Invalid or missing stateId from key:`,JSON.stringify(t));return}let a=u.node()?.querySelectorAll(`g.node, g.rough-node`),o;if(a?.forEach(e=>{let t=e.textContent?.trim();(e.id===i?.domId||t===r)&&(o=e)}),!o){n.warn(`⚠️ Could not find node matching text:`,r);return}let s=o.parentNode;if(!s){n.warn(`⚠️ Node has no parent, cannot wrap:`,r);return}let c=document.createElementNS(`http://www.w3.org/2000/svg`,`a`),d=e.url.replace(/^"+|"+$/g,``);if(c.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,d),c.setAttribute(`target`,`_blank`),e.tooltip){let t=e.tooltip.replace(/^"+|"+$/g,``);c.setAttribute(`title`,t),o.setAttribute(`title`,t)}s.replaceChild(c,o),c.appendChild(o),n.info(`🔗 Wrapped node in <a> tag for:`,r,e.url)})}catch(e){n.error(`❌ Error injecting clickable links:`,e)}m.insertTitle(u,`statediagramTitleText`,s?.titleTopMargin??25,a.db.getDiagramTitle()),_(u,8,z,s?.useMaxWidth??!0)},`draw`),getDir:W},G=new Map,K=0;function q(e=``,t=0,n=``,r=H){return`${me}-${e}${n!==null&&n.length>0?`${r}${n}`:``}-${t}`}y(q,`stateDomId`),t(q,`stateDomId`);var _e=t((e,t,i,a,o,s,c,l)=>{n.trace(`items`,t),t.forEach(t=>{switch(t.stmt){case T:Z(e,t,i,a,o,s,c,l);break;case j:Z(e,t,i,a,o,s,c,l);break;case D:{Z(e,t.state1,i,a,o,s,c,l),Z(e,t.state2,i,a,o,s,c,l);let n=c===`neo`,d={id:`edge`+K,start:t.state1.id,end:t.state2.id,arrowhead:`normal`,arrowTypeEnd:n?`arrow_barb_neo`:`arrow_barb`,style:N,labelStyle:``,label:u.sanitizeText(t.description??``,r()),arrowheadStyle:P,labelpos:F,labelType:I,thickness:ee,classes:ce,look:c};o.push(d),K++}}})},`setupDoc`),ve=t((e,t=C)=>{let n=t;if(e.doc)for(let t of e.doc)t.stmt===`dir`&&(n=t.value);return n},`getDir`);function J(e,t,n){if(!t.id||t.id===`</join></fork>`||t.id===`</choice>`)return;t.cssClasses&&(Array.isArray(t.cssCompiledStyles)||(t.cssCompiledStyles=[]),t.cssClasses.split(` `).forEach(e=>{let r=n.get(e);r&&(t.cssCompiledStyles=[...t.cssCompiledStyles??[],...r.styles])}));let r=e.find(e=>e.id===t.id);r?Object.assign(r,t):e.push(t)}y(J,`insertOrUpdateNode`),t(J,`insertOrUpdateNode`);function Y(e){return e?.classes?.join(` `)??``}y(Y,`getClassesFromDbInfo`),t(Y,`getClassesFromDbInfo`);function X(e){return e?.styles??[]}y(X,`getStylesFromDbInfo`),t(X,`getStylesFromDbInfo`);var Z=t((e,t,i,a,o,s,c,l)=>{let d=t.id,f=i.get(d),p=Y(f),m=X(f),h=r();if(n.info(`dataFetcher parsedItem`,t,f,m),d!==`root`){let r=L;t.start===!0?r=te:t.start===!1&&(r=ne),t.type!==j&&(r=t.type),G.get(d)||G.set(d,{id:d,shape:r,description:u.sanitizeText(d,h),cssClasses:`${p} ${se}`,cssStyles:m});let i=G.get(d);t.description&&(Array.isArray(i.description)?(i.shape=R,i.description.push(t.description)):i.description?.length&&i.description.length>0?(i.shape=R,i.description=i.description===d?[t.description]:[i.description,t.description]):(i.shape=L,i.description=t.description),i.description=u.sanitizeTextOrArray(i.description,h)),i.description?.length===1&&i.shape===R&&(i.shape=i.type===`group`?ie:L),!i.type&&t.doc&&(n.info(`Setting cluster for XCX`,d,ve(t)),i.type=`group`,i.isGroup=!0,i.dir=ve(t),i.shape=t.type===M?re:ie,i.cssClasses=`${i.cssClasses} ${fe} ${s?pe:``}`);let f={labelStyle:``,shape:i.shape,label:i.description,cssClasses:i.cssClasses,cssCompiledStyles:[],cssStyles:i.cssStyles,id:d,dir:i.dir,domId:q(d,K),type:i.type,isGroup:i.type===`group`,padding:8,rx:10,ry:10,look:c,labelType:`markdown`};if(f.shape===re&&(f.label=``),e&&e.id!==`root`&&(n.trace(`Setting node `,d,` to be child of its parent `,e.id),f.parentId=e.id),f.centerLabel=!0,t.note){let e={labelStyle:``,shape:ae,label:t.note.text,labelType:`markdown`,cssClasses:de,cssStyles:[],cssCompiledStyles:[],id:d+he+`-`+K,domId:q(d,K,V),type:i.type,isGroup:i.type===`group`,padding:h.flowchart?.padding,look:c,position:t.note.position},n=d+U,r={labelStyle:``,shape:oe,label:t.note.text,cssClasses:i.cssClasses,cssStyles:[],id:d+U,domId:q(d,K,B),type:`group`,isGroup:!0,padding:16,look:c,position:t.note.position};K++,r.id=n,e.parentId=n,J(a,r,l),J(a,e,l),J(a,f,l);let s=d,u=e.id;t.note.position===`left of`&&(s=e.id,u=d),o.push({id:s+`-`+u,start:s,end:u,arrowhead:`none`,arrowTypeEnd:``,style:N,labelStyle:``,classes:ue,arrowheadStyle:P,labelpos:F,labelType:I,thickness:ee,look:c})}else J(a,f,l)}t.doc&&(n.trace(`Adding nodes children `),_e(t,t.doc,i,a,o,!s,c,l))},`dataFetcher`),ye=t(()=>{G.clear(),K=0},`reset`),Q={START_NODE:`[*]`,START_TYPE:`start`,END_NODE:`[*]`,END_TYPE:`end`,COLOR_KEYWORD:`color`,FILL_KEYWORD:`fill`,BG_FILL:`bgFill`,STYLECLASS_SEP:`,`},be=t(()=>new Map,`newClassesList`),xe=t(()=>({relations:[],states:new Map,documents:{}}),`newDoc`),$=t(e=>JSON.parse(JSON.stringify(e)),`clone`),Se=class{static{y(this,`StateDB`)}constructor(e){this.version=e,this.nodes=[],this.edges=[],this.rootDoc=[],this.classes=be(),this.documents={root:xe()},this.currentDocument=this.documents.root,this.startEndCount=0,this.dividerCnt=0,this.links=new Map,this.funs=[],this.getAccTitle=f,this.setAccTitle=c,this.getAccDescription=l,this.setAccDescription=a,this.setDiagramTitle=s,this.getDiagramTitle=i,this.clear(),this.setRootDoc=this.setRootDoc.bind(this),this.getDividerId=this.getDividerId.bind(this),this.setDirection=this.setDirection.bind(this),this.trimColon=this.trimColon.bind(this),this.bindFunctions=this.bindFunctions.bind(this)}static{t(this,`StateDB`)}static{this.relationType={AGGREGATION:0,EXTENSION:1,COMPOSITION:2,DEPENDENCY:3}}extract(e){this.clear(!0);for(let t of Array.isArray(e)?e:e.doc)switch(t.stmt){case T:this.addState(t.id.trim(),t.type,t.doc,t.description,t.note);break;case D:this.addRelation(t.state1,t.state2,t.description);break;case O:this.addStyleClass(t.id.trim(),t.classes);break;case k:this.handleStyleDef(t);break;case A:this.setCssClass(t.id.trim(),t.styleClass);break;case`click`:this.addLink(t.id,t.url,t.tooltip)}let t=this.getStates(),n=r();ye(),Z(void 0,this.getRootDocV2(),t,this.nodes,this.edges,!0,n.look,this.classes);for(let e of this.nodes)if(Array.isArray(e.label)){if(e.description=e.label.slice(1),e.isGroup&&e.description.length>0)throw Error(`Group nodes can only have label. Remove the additional description for node [${e.id}]`);e.label=e.label[0]}}handleStyleDef(e){let t=e.id.trim().split(`,`),n=e.styleClass.split(`,`);for(let e of t){let t=this.getState(e);if(!t){let n=e.trim();this.addState(n),t=this.getState(n)}t&&(t.styles=n.map(e=>e.replace(/;/g,``)?.trim()))}}setRootDoc(e){n.info(`Setting root doc`,e),this.rootDoc=e,this.version===1?this.extract(e):this.extract(this.getRootDocV2())}docTranslator(e,t,n){if(t.stmt===D){this.docTranslator(e,t.state1,!0),this.docTranslator(e,t.state2,!1);return}if(t.stmt===T&&(t.id===Q.START_NODE?(t.id=e.id+(n?`_start`:`_end`),t.start=n):t.id=t.id.trim()),t.stmt!==E&&t.stmt!==T||!t.doc)return;let r=[],i=[];for(let e of t.doc)if(e.type===M){let t=$(e);t.doc=$(i),r.push(t),i=[]}else i.push(e);if(r.length>0&&i.length>0){let e={stmt:T,id:p(),type:`divider`,doc:$(i)};r.push($(e)),t.doc=r}t.doc.forEach(e=>this.docTranslator(t,e,!0))}getRootDocV2(){return this.docTranslator({id:E,stmt:E},{id:E,stmt:E,doc:this.rootDoc},!0),{id:E,doc:this.rootDoc}}addState(e,t=j,i=void 0,a=void 0,o=void 0,s=void 0,c=void 0,l=void 0){let d=e?.trim();if(!this.currentDocument.states.has(d))n.info(`Adding state `,d,a),this.currentDocument.states.set(d,{stmt:T,id:d,descriptions:[],type:t,doc:i,note:o,classes:[],styles:[],textStyles:[]});else{let e=this.currentDocument.states.get(d);if(!e)throw Error(`State not found: ${d}`);e.doc||=i,e.type||=t}if(a&&(n.info(`Setting state description`,d,a),(Array.isArray(a)?a:[a]).forEach(e=>this.addDescription(d,e.trim()))),o){let e=this.currentDocument.states.get(d);if(!e)throw Error(`State not found: ${d}`);e.note=o,e.note.text=u.sanitizeText(e.note.text,r())}s&&(n.info(`Setting state classes`,d,s),(Array.isArray(s)?s:[s]).forEach(e=>this.setCssClass(d,e.trim()))),c&&(n.info(`Setting state styles`,d,c),(Array.isArray(c)?c:[c]).forEach(e=>this.setStyle(d,e.trim()))),l&&(n.info(`Setting state styles`,d,c),(Array.isArray(l)?l:[l]).forEach(e=>this.setTextStyle(d,e.trim())))}clear(e){this.nodes=[],this.edges=[],this.funs=[this.setupToolTips.bind(this)],this.documents={root:xe()},this.currentDocument=this.documents.root,this.startEndCount=0,this.classes=be(),e||(this.links=new Map,d())}getState(e){return this.currentDocument.states.get(e)}getStates(){return this.currentDocument.states}logDocuments(){n.info(`Documents = `,this.documents)}getRelations(){return this.currentDocument.relations}addLink(e,t,r){this.links.set(e,{url:t,tooltip:r}),n.warn(`Adding link`,e,t,r)}getLinks(){return this.links}startIdIfNeeded(e=``){return e===Q.START_NODE?(this.startEndCount++,`${Q.START_TYPE}${this.startEndCount}`):e}startTypeIfNeeded(e=``,t=j){return e===Q.START_NODE?Q.START_TYPE:t}endIdIfNeeded(e=``){return e===Q.END_NODE?(this.startEndCount++,`${Q.END_TYPE}${this.startEndCount}`):e}endTypeIfNeeded(e=``,t=j){return e===Q.END_NODE?Q.END_TYPE:t}addRelationObjs(e,t,n=``){let i=this.startIdIfNeeded(e.id.trim()),a=this.startTypeIfNeeded(e.id.trim(),e.type),o=this.startIdIfNeeded(t.id.trim()),s=this.startTypeIfNeeded(t.id.trim(),t.type);this.addState(i,a,e.doc,e.description,e.note,e.classes,e.styles,e.textStyles),this.addState(o,s,t.doc,t.description,t.note,t.classes,t.styles,t.textStyles),this.currentDocument.relations.push({id1:i,id2:o,relationTitle:u.sanitizeText(n,r())})}addRelation(e,t,n){if(typeof e==`object`&&typeof t==`object`)this.addRelationObjs(e,t,n);else if(typeof e==`string`&&typeof t==`string`){let i=this.startIdIfNeeded(e.trim()),a=this.startTypeIfNeeded(e),o=this.endIdIfNeeded(t.trim()),s=this.endTypeIfNeeded(t);this.addState(i,a),this.addState(o,s),this.currentDocument.relations.push({id1:i,id2:o,relationTitle:n?u.sanitizeText(n,r()):void 0})}}addDescription(e,t){let n=this.currentDocument.states.get(e),i=t.startsWith(`:`)?t.replace(`:`,``).trim():t;n?.descriptions?.push(u.sanitizeText(i,r()))}cleanupLabel(e){return e.startsWith(`:`)?e.slice(2).trim():e.trim()}getDividerId(){return this.dividerCnt++,`divider-id-${this.dividerCnt}`}addStyleClass(e,t=``){this.classes.has(e)||this.classes.set(e,{id:e,styles:[],textStyles:[]});let n=this.classes.get(e);t&&n&&t.split(Q.STYLECLASS_SEP).forEach(e=>{let t=e.replace(/([^;]*);/,`$1`).trim();if(RegExp(Q.COLOR_KEYWORD).exec(e)){let e=t.replace(Q.FILL_KEYWORD,Q.BG_FILL).replace(Q.COLOR_KEYWORD,Q.FILL_KEYWORD);n.textStyles.push(e)}n.styles.push(t)})}getClasses(){return this.classes}setupToolTips(t){let n=h();e(t).select(`svg`).selectAll(`g.node, g.rough-node`).on(`mouseover`,t=>{let r=e(t.currentTarget),i=r.attr(`title`);if(i===null)return;let a=t.currentTarget?.getBoundingClientRect();n.transition().duration(200).style(`opacity`,`.9`),n.style(`left`,window.scrollX+a.left+(a.right-a.left)/2+`px`).style(`top`,window.scrollY+a.bottom+`px`),n.html(o.sanitize(i)),r.classed(`hover`,!0)}).on(`mouseout`,t=>{n.transition().duration(500).style(`opacity`,0),e(t.currentTarget).classed(`hover`,!1)})}setCssClass(e,t){e.split(`,`).forEach(e=>{let n=this.getState(e);if(!n){let t=e.trim();this.addState(t),n=this.getState(t)}n?.classes?.push(t)})}setStyle(e,t){this.getState(e)?.styles?.push(t)}setTextStyle(e,t){this.getState(e)?.textStyles?.push(t)}bindFunctions(e){this.funs.forEach(t=>{t(e)})}getDirectionStatement(){return this.rootDoc.find(e=>e.stmt===w)}getDirection(){return this.getDirectionStatement()?.value??S}setDirection(e){let t=this.getDirectionStatement();t?t.value=e:this.rootDoc.unshift({stmt:w,value:e})}trimColon(e){return e.startsWith(`:`)?e.slice(1).trim():e.trim()}getData(){let e=r();return{nodes:this.nodes,edges:this.edges,other:{},config:e,direction:W(this.getRootDocV2())}}getConfig(){return r().state}},Ce=t(e=>`
defs [id$="-barbEnd"] {
    fill: ${e.transitionColor};
    stroke: ${e.transitionColor};
  }
g.stateGroup text {
  fill: ${e.nodeBorder};
  stroke: none;
  font-size: 10px;
}
g.stateGroup text {
  fill: ${e.textColor};
  stroke: none;
  font-size: 10px;

}
g.stateGroup .state-title {
  font-weight: bolder;
  fill: ${e.stateLabelColor};
}

g.stateGroup rect {
  fill: ${e.mainBkg};
  stroke: ${e.nodeBorder};
}

g.stateGroup line {
  stroke: ${e.lineColor};
  stroke-width: ${e.strokeWidth||1};
}

.transition {
  stroke: ${e.transitionColor};
  stroke-width: ${e.strokeWidth||1};
  fill: none;
}

.stateGroup .composit {
  fill: ${e.background};
  border-bottom: 1px
}

.stateGroup .alt-composit {
  fill: #e0e0e0;
  border-bottom: 1px
}

.state-note {
  stroke: ${e.noteBorderColor};
  fill: ${e.noteBkgColor};

  text {
    fill: ${e.noteTextColor};
    stroke: none;
    font-size: 10px;
  }
}

.stateLabel .box {
  stroke: none;
  stroke-width: 0;
  fill: ${e.mainBkg};
  opacity: 0.5;
}

.edgeLabel .label rect {
  fill: ${e.labelBackgroundColor};
  opacity: 0.5;
}
.edgeLabel {
  background-color: ${e.edgeLabelBackground};
  p {
    background-color: ${e.edgeLabelBackground};
  }
  rect {
    opacity: 0.5;
    background-color: ${e.edgeLabelBackground};
    fill: ${e.edgeLabelBackground};
  }
  text-align: center;
}
.edgeLabel .label text {
  fill: ${e.transitionLabelColor||e.tertiaryTextColor};
}
.label div .edgeLabel {
  color: ${e.transitionLabelColor||e.tertiaryTextColor};
}

.stateLabel text {
  fill: ${e.stateLabelColor};
  font-size: 10px;
  font-weight: bold;
}

.node circle.state-start {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node .fork-join {
  fill: ${e.specialStateColor};
  stroke: ${e.specialStateColor};
}

.node circle.state-end {
  fill: ${e.innerEndBackground};
  stroke: ${e.background};
  stroke-width: 1.5
}
.end-state-inner {
  fill: ${e.compositeBackground||e.background};
  // stroke: ${e.background};
  stroke-width: 1.5
}

.node rect {
  fill: ${e.stateBkg||e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: ${e.strokeWidth||1}px;
}
.node polygon {
  fill: ${e.mainBkg};
  stroke: ${e.stateBorder||e.nodeBorder};;
  stroke-width: ${e.strokeWidth||1}px;
}
[id$="-barbEnd"] {
  fill: ${e.lineColor};
}

.statediagram-cluster rect {
  fill: ${e.compositeTitleBackground};
  stroke: ${e.stateBorder||e.nodeBorder};
  stroke-width: ${e.strokeWidth||1}px;
}

.cluster-label, .nodeLabel {
  color: ${e.stateLabelColor};
  // line-height: 1;
}

.statediagram-cluster rect.outer {
  rx: 5px;
  ry: 5px;
}
.statediagram-state .divider {
  stroke: ${e.stateBorder||e.nodeBorder};
}

.statediagram-state .title-state {
  rx: 5px;
  ry: 5px;
}
.statediagram-cluster.statediagram-cluster .inner {
  fill: ${e.compositeBackground||e.background};
}
.statediagram-cluster.statediagram-cluster-alt .inner {
  fill: ${e.altBackground?e.altBackground:`#efefef`};
}

.statediagram-cluster .inner {
  rx:0;
  ry:0;
}

.statediagram-state rect.basic {
  rx: 5px;
  ry: 5px;
}
.statediagram-state rect.divider {
  stroke-dasharray: 10,10;
  fill: ${e.altBackground?e.altBackground:`#efefef`};
}

.note-edge {
  stroke-dasharray: 5;
}

.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}
.statediagram-note rect {
  fill: ${e.noteBkgColor};
  stroke: ${e.noteBorderColor};
  stroke-width: 1px;
  rx: 0;
  ry: 0;
}

.statediagram-note text {
  fill: ${e.noteTextColor};
}

.statediagram-note .nodeLabel {
  color: ${e.noteTextColor};
}
.statediagram .edgeLabel {
  color: red; // ${e.noteTextColor};
}

[id$="-dependencyStart"], [id$="-dependencyEnd"] {
  fill: ${e.lineColor};
  stroke: ${e.lineColor};
  stroke-width: ${e.strokeWidth||1};
}

.statediagramTitleText {
  text-anchor: middle;
  font-size: 18px;
  fill: ${e.textColor};
}

[data-look="neo"].statediagram-cluster rect {
  fill: ${e.mainBkg};
  stroke: ${e.useGradient?`url(`+e.svgId+`-gradient)`:e.stateBorder||e.nodeBorder};
  stroke-width: ${e.strokeWidth??1};
}
[data-look="neo"].statediagram-cluster rect.outer {
  rx: ${e.radius}px;
  ry: ${e.radius}px;
  filter: ${e.dropShadow?e.dropShadow.replace(`url(#drop-shadow)`,`url(${e.svgId}-drop-shadow)`):`none`}
}
`,`getStyles`);export{Ce as i,x as n,ge as r,Se as t};