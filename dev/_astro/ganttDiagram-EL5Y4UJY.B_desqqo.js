import{X as e,Z as t,et as n}from"./src.B2_BndWA.js";import{t as r}from"./linear.Cbi17xCf.js";import{_ as i,a,b as o,c as s,d as c,f as l,g as u,h as d,i as f,l as p,m,n as h,o as g,p as _,r as v,s as y,t as b,u as x,v as S,y as C}from"./time.BOHsYPil.js";import{n as w}from"./chunk-Y2CYZVJY.DsF7k-Jl.js";import{r as T,t as E}from"./chunk-X3CZISLH.CmQIif5t.js";import{H as D,K as O,U as k,a as A,c as j,s as ee,v as M,w as N,x as P,y as F}from"./chunk-DU6HZSFF.BoLJipnC.js";import{t as te}from"./dist.OWNJFjib.js";import{g as ne}from"./chunk-75Z2AOVW.DMe2hltC.js";var re=t(((e,t,n)=>{(function(n,r){typeof e==`object`&&t!==void 0?t.exports=r():typeof define==`function`&&define.amd?define(r):(n=typeof globalThis<`u`?globalThis:n||self).dayjs_plugin_isoWeek=r()})(n,(function(){return function(e,t,n){var r=function(e){return e.add(4-e.isoWeekday(),`day`)},i=t.prototype;i.isoWeekYear=function(){return r(this).year()},i.isoWeek=function(e){if(!this.$utils().u(e))return this.add(7*(e-this.isoWeek()),`day`);var t,i,a,o,s=r(this),c=(t=this.isoWeekYear(),i=this.$u,a=(i?n.utc:n)().year(t).startOf(`year`),o=4-a.isoWeekday(),a.isoWeekday()>4&&(o+=7),a.add(o,`day`));return s.diff(c,`week`)+1},i.isoWeekday=function(e){return this.$utils().u(e)?this.day()||7:this.day(this.day()%7?e:e-7)};var a=i.startOf;i.startOf=function(e,t){var n=this.$utils(),r=!!n.u(t)||t;return n.p(e)===`isoweek`?r?this.date(this.date()-(this.isoWeekday()-1)).startOf(`day`):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf(`day`):a.bind(this)(e,t)}}}))})),ie=t(((e,t,n)=>{(function(n,r){typeof e==`object`&&t!==void 0?t.exports=r():typeof define==`function`&&define.amd?define(r):(n=typeof globalThis<`u`?globalThis:n||self).dayjs_plugin_customParseFormat=r()})(n,(function(){var e={LTS:`h:mm:ss A`,LT:`h:mm A`,L:`MM/DD/YYYY`,LL:`MMMM D, YYYY`,LLL:`MMMM D, YYYY h:mm A`,LLLL:`dddd, MMMM D, YYYY h:mm A`},t=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,n=/\d/,r=/\d\d/,i=/\d\d?/,a=/\d*[^-_:/,()\s\d]+/,o={},s=function(e){return(e=+e)+(e>68?1900:2e3)},c=function(e){return function(t){this[e]=+t}},l=[/[+-]\d\d:?(\d\d)?|Z/,function(e){(this.zone||={}).offset=function(e){if(!e||e===`Z`)return 0;var t=e.match(/([+-]|\d\d)/g),n=60*t[1]+(+t[2]||0);return n===0?0:t[0]===`+`?-n:n}(e)}],u=function(e){var t=o[e];return t&&(t.indexOf?t:t.s.concat(t.f))},d=function(e,t){var n,r=o.meridiem;if(r){for(var i=1;i<=24;i+=1)if(e.indexOf(r(i,0,t))>-1){n=i>12;break}}else n=e===(t?`pm`:`PM`);return n},f={A:[a,function(e){this.afternoon=d(e,!1)}],a:[a,function(e){this.afternoon=d(e,!0)}],Q:[n,function(e){this.month=3*(e-1)+1}],S:[n,function(e){this.milliseconds=100*e}],SS:[r,function(e){this.milliseconds=10*e}],SSS:[/\d{3}/,function(e){this.milliseconds=+e}],s:[i,c(`seconds`)],ss:[i,c(`seconds`)],m:[i,c(`minutes`)],mm:[i,c(`minutes`)],H:[i,c(`hours`)],h:[i,c(`hours`)],HH:[i,c(`hours`)],hh:[i,c(`hours`)],D:[i,c(`day`)],DD:[r,c(`day`)],Do:[a,function(e){var t=o.ordinal,n=e.match(/\d+/);if(this.day=n[0],t)for(var r=1;r<=31;r+=1)t(r).replace(/\[|\]/g,``)===e&&(this.day=r)}],w:[i,c(`week`)],ww:[r,c(`week`)],M:[i,c(`month`)],MM:[r,c(`month`)],MMM:[a,function(e){var t=u(`months`),n=(u(`monthsShort`)||t.map((function(e){return e.slice(0,3)}))).indexOf(e)+1;if(n<1)throw Error();this.month=n%12||n}],MMMM:[a,function(e){var t=u(`months`).indexOf(e)+1;if(t<1)throw Error();this.month=t%12||t}],Y:[/[+-]?\d+/,c(`year`)],YY:[r,function(e){this.year=s(e)}],YYYY:[/\d{4}/,c(`year`)],Z:l,ZZ:l};function p(n){for(var r=n,i=o&&o.formats,a=(n=r.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(t,n,r){var a=r&&r.toUpperCase();return n||i[r]||e[r]||i[a].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(e,t,n){return t||n.slice(1)}))}))).match(t),s=a.length,c=0;c<s;c+=1){var l=a[c],u=f[l],d=u&&u[0],p=u&&u[1];a[c]=p?{regex:d,parser:p}:l.replace(/^\[|\]$/g,``)}return function(e){for(var t={},n=0,r=0;n<s;n+=1){var i=a[n];if(typeof i==`string`)r+=i.length;else{var o=i.regex,c=i.parser,l=e.slice(r),u=o.exec(l)[0];c.call(t,u),e=e.replace(u,``)}}return function(e){var t=e.afternoon;if(t!==void 0){var n=e.hours;t?n<12&&(e.hours+=12):n===12&&(e.hours=0),delete e.afternoon}}(t),t}}return function(e,t,n){n.p.customParseFormat=!0,e&&e.parseTwoDigitYear&&(s=e.parseTwoDigitYear);var r=t.prototype,i=r.parse;r.parse=function(e){var t=e.date,r=e.utc,a=e.args;this.$u=r;var s=a[1];if(typeof s==`string`){var c=!0===a[2],l=!0===a[3],u=c||l,d=a[2];l&&(d=a[2]),o=this.$locale(),!c&&d&&(o=n.Ls[d]),this.$d=function(e,t,n,r){try{if([`x`,`X`].indexOf(t)>-1)return new Date((t===`X`?1e3:1)*e);var i=p(t)(e),a=i.year,o=i.month,s=i.day,c=i.hours,l=i.minutes,u=i.seconds,d=i.milliseconds,f=i.zone,m=i.week,h=new Date,g=s||(a||o?1:h.getDate()),_=a||h.getFullYear(),v=0;a&&!o||(v=o>0?o-1:h.getMonth());var y,b=c||0,x=l||0,S=u||0,C=d||0;return f?new Date(Date.UTC(_,v,g,b,x,S,C+60*f.offset*1e3)):n?new Date(Date.UTC(_,v,g,b,x,S,C)):(y=new Date(_,v,g,b,x,S,C),m&&(y=r(y).week(m).toDate()),y)}catch{return new Date(``)}}(t,s,r,n),this.init(),d&&!0!==d&&(this.$L=this.locale(d).$L),u&&t!=this.format(s)&&(this.$d=new Date(``)),o={}}else if(s instanceof Array)for(var f=s.length,m=1;m<=f;m+=1){a[1]=s[m-1];var h=n.apply(this,a);if(h.isValid()){this.$d=h.$d,this.$L=h.$L,this.init();break}m===f&&(this.$d=new Date(``))}else i.call(this,e)}}}))})),I=t(((e,t,n)=>{(function(n,r){typeof e==`object`&&t!==void 0?t.exports=r():typeof define==`function`&&define.amd?define(r):(n=typeof globalThis<`u`?globalThis:n||self).dayjs_plugin_advancedFormat=r()})(n,(function(){return function(e,t){var n=t.prototype,r=n.format;n.format=function(e){var t=this,n=this.$locale();if(!this.isValid())return r.bind(this)(e);var i=this.$utils(),a=(e||`YYYY-MM-DDTHH:mm:ssZ`).replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(e){switch(e){case`Q`:return Math.ceil((t.$M+1)/3);case`Do`:return n.ordinal(t.$D);case`gggg`:return t.weekYear();case`GGGG`:return t.isoWeekYear();case`wo`:return n.ordinal(t.week(),`W`);case`w`:case`ww`:return i.s(t.week(),e===`w`?1:2,`0`);case`W`:case`WW`:return i.s(t.isoWeek(),e===`W`?1:2,`0`);case`k`:case`kk`:return i.s(String(t.$H===0?24:t.$H),e===`k`?1:2,`0`);case`X`:return Math.floor(t.$d.getTime()/1e3);case`x`:return t.$d.getTime();case`z`:return`[`+t.offsetName()+`]`;case`zzz`:return`[`+t.offsetName(`long`)+`]`;default:return e}}));return r.bind(this)(a)}}}))})),L=t(((e,t,n)=>{(function(n,r){typeof e==`object`&&t!==void 0?t.exports=r():typeof define==`function`&&define.amd?define(r):(n=typeof globalThis<`u`?globalThis:n||self).dayjs_plugin_duration=r()})(n,(function(){var e,t,n=1e3,r=6e4,i=36e5,a=864e5,o=31536e6,s=2628e6,c=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,l=/\[([^\]]+)]|YYYY|YY|Y|M{1,2}|D{1,2}|H{1,2}|m{1,2}|s{1,2}|SSS/g,u={years:o,months:s,days:a,hours:i,minutes:r,seconds:n,milliseconds:1,weeks:6048e5},d=function(e){return e instanceof v},f=function(e,t,n){return new v(e,n,t.$l)},p=function(e){return t.p(e)+`s`},m=function(e){return e<0},h=function(e){return m(e)?Math.ceil(e):Math.floor(e)},g=function(e){return Math.abs(e)},_=function(e,t){return e?m(e)?{negative:!0,format:``+g(e)+t}:{negative:!1,format:``+e+t}:{negative:!1,format:``}},v=function(){function m(e,t,n){var r=this;if(this.$d={},this.$l=n,e===void 0&&(this.$ms=0,this.parseFromMilliseconds()),t)return f(e*u[p(t)],this);if(typeof e==`number`)return this.$ms=e,this.parseFromMilliseconds(),this;if(typeof e==`object`)return Object.keys(e).forEach((function(t){r.$d[p(t)]=e[t]})),this.calMilliseconds(),this;if(typeof e==`string`){var i=e.match(c);if(i){var a=i.slice(2).map((function(e){return e==null?0:Number(e)}));return this.$d.years=a[0],this.$d.months=a[1],this.$d.weeks=a[2],this.$d.days=a[3],this.$d.hours=a[4],this.$d.minutes=a[5],this.$d.seconds=a[6],this.calMilliseconds(),this}}return this}var g=m.prototype;return g.calMilliseconds=function(){var e=this;this.$ms=Object.keys(this.$d).reduce((function(t,n){return t+(e.$d[n]||0)*u[n]}),0)},g.parseFromMilliseconds=function(){var e=this.$ms;this.$d.years=h(e/o),e%=o,this.$d.months=h(e/s),e%=s,this.$d.days=h(e/a),e%=a,this.$d.hours=h(e/i),e%=i,this.$d.minutes=h(e/r),e%=r,this.$d.seconds=h(e/n),e%=n,this.$d.milliseconds=e},g.toISOString=function(){var e=_(this.$d.years,`Y`),t=_(this.$d.months,`M`),n=+this.$d.days||0;this.$d.weeks&&(n+=7*this.$d.weeks);var r=_(n,`D`),i=_(this.$d.hours,`H`),a=_(this.$d.minutes,`M`),o=this.$d.seconds||0;this.$d.milliseconds&&(o+=this.$d.milliseconds/1e3,o=Math.round(1e3*o)/1e3);var s=_(o,`S`),c=e.negative||t.negative||r.negative||i.negative||a.negative||s.negative,l=i.format||a.format||s.format?`T`:``,u=(c?`-`:``)+`P`+e.format+t.format+r.format+l+i.format+a.format+s.format;return u===`P`||u===`-P`?`P0D`:u},g.toJSON=function(){return this.toISOString()},g.format=function(e){var n=e||`YYYY-MM-DDTHH:mm:ss`,r={Y:this.$d.years,YY:t.s(this.$d.years,2,`0`),YYYY:t.s(this.$d.years,4,`0`),M:this.$d.months,MM:t.s(this.$d.months,2,`0`),D:this.$d.days,DD:t.s(this.$d.days,2,`0`),H:this.$d.hours,HH:t.s(this.$d.hours,2,`0`),m:this.$d.minutes,mm:t.s(this.$d.minutes,2,`0`),s:this.$d.seconds,ss:t.s(this.$d.seconds,2,`0`),SSS:t.s(this.$d.milliseconds,3,`0`)};return n.replace(l,(function(e,t){return t||String(r[e])}))},g.as=function(e){return this.$ms/u[p(e)]},g.get=function(e){var t=this.$ms,n=p(e);return n===`milliseconds`?t%=1e3:t=n===`weeks`?h(t/u[n]):this.$d[n],t||0},g.add=function(e,t,n){var r=t?e*u[p(t)]:d(e)?e.$ms:f(e,this).$ms;return f(this.$ms+r*(n?-1:1),this)},g.subtract=function(e,t){return this.add(e,t,!0)},g.locale=function(e){var t=this.clone();return t.$l=e,t},g.clone=function(){return f(this.$ms,this)},g.humanize=function(t){return e().add(this.$ms,`ms`).locale(this.$l).fromNow(!t)},g.valueOf=function(){return this.asMilliseconds()},g.milliseconds=function(){return this.get(`milliseconds`)},g.asMilliseconds=function(){return this.as(`milliseconds`)},g.seconds=function(){return this.get(`seconds`)},g.asSeconds=function(){return this.as(`seconds`)},g.minutes=function(){return this.get(`minutes`)},g.asMinutes=function(){return this.as(`minutes`)},g.hours=function(){return this.get(`hours`)},g.asHours=function(){return this.as(`hours`)},g.days=function(){return this.get(`days`)},g.asDays=function(){return this.as(`days`)},g.weeks=function(){return this.get(`weeks`)},g.asWeeks=function(){return this.as(`weeks`)},g.months=function(){return this.get(`months`)},g.asMonths=function(){return this.as(`months`)},g.years=function(){return this.get(`years`)},g.asYears=function(){return this.as(`years`)},m}(),y=function(e,t,n){return e.add(t.years()*n,`y`).add(t.months()*n,`M`).add(t.days()*n,`d`).add(t.hours()*n,`h`).add(t.minutes()*n,`m`).add(t.seconds()*n,`s`).add(t.milliseconds()*n,`ms`)};return function(n,r,i){e=i,t=i().$utils(),i.duration=function(e,t){return f(e,{$l:i.locale()},t)},i.isDuration=d;var a=r.prototype.add,o=r.prototype.subtract;r.prototype.add=function(e,t){return d(e)?y(this,e,1):a.bind(this)(e,t)},r.prototype.subtract=function(e,t){return d(e)?y(this,e,-1):o.bind(this)(e,t)}}}))})),ae=te(),R=n(T(),1),oe=n(re(),1),se=n(ie(),1),ce=n(I(),1),le=n(L(),1),ue=(function(){var e=w(function(e,t,n,r){for(n||={},r=e.length;r--;n[e[r]]=t);return n},`o`),t=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],n=[1,26],r=[1,27],i=[1,28],a=[1,29],o=[1,30],s=[1,31],c=[1,32],l=[1,33],u=[1,34],d=[1,9],f=[1,10],p=[1,11],m=[1,12],h=[1,13],g=[1,14],_=[1,15],v=[1,16],y=[1,19],b=[1,20],x=[1,21],S=[1,22],C=[1,23],T=[1,25],E=[1,35],D={trace:w(function(){},`trace`),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:`error`,4:`gantt`,6:`EOF`,8:`SPACE`,10:`NL`,12:`weekday_monday`,13:`weekday_tuesday`,14:`weekday_wednesday`,15:`weekday_thursday`,16:`weekday_friday`,17:`weekday_saturday`,18:`weekday_sunday`,20:`weekend_friday`,21:`weekend_saturday`,22:`dateFormat`,23:`inclusiveEndDates`,24:`topAxis`,25:`axisFormat`,26:`tickInterval`,27:`excludes`,28:`includes`,29:`todayMarker`,30:`title`,31:`acc_title`,32:`acc_title_value`,33:`acc_descr`,34:`acc_descr_value`,35:`acc_descr_multiline_value`,36:`section`,38:`taskTxt`,39:`taskData`,40:`click`,41:`callbackname`,42:`callbackargs`,43:`href`},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:w(function(e,t,n,r,i,a,o){var s=a.length-1;switch(i){case 1:return a[s-1];case 2:this.$=[];break;case 3:a[s-1].push(a[s]),this.$=a[s-1];break;case 4:case 5:this.$=a[s];break;case 6:case 7:this.$=[];break;case 8:r.setWeekday(`monday`);break;case 9:r.setWeekday(`tuesday`);break;case 10:r.setWeekday(`wednesday`);break;case 11:r.setWeekday(`thursday`);break;case 12:r.setWeekday(`friday`);break;case 13:r.setWeekday(`saturday`);break;case 14:r.setWeekday(`sunday`);break;case 15:r.setWeekend(`friday`);break;case 16:r.setWeekend(`saturday`);break;case 17:r.setDateFormat(a[s].substr(11)),this.$=a[s].substr(11);break;case 18:r.enableInclusiveEndDates(),this.$=a[s].substr(18);break;case 19:r.TopAxis(),this.$=a[s].substr(8);break;case 20:r.setAxisFormat(a[s].substr(11)),this.$=a[s].substr(11);break;case 21:r.setTickInterval(a[s].substr(13)),this.$=a[s].substr(13);break;case 22:r.setExcludes(a[s].substr(9)),this.$=a[s].substr(9);break;case 23:r.setIncludes(a[s].substr(9)),this.$=a[s].substr(9);break;case 24:r.setTodayMarker(a[s].substr(12)),this.$=a[s].substr(12);break;case 27:r.setDiagramTitle(a[s].substr(6)),this.$=a[s].substr(6);break;case 28:this.$=a[s].trim(),r.setAccTitle(this.$);break;case 29:case 30:this.$=a[s].trim(),r.setAccDescription(this.$);break;case 31:r.addSection(a[s].substr(8)),this.$=a[s].substr(8);break;case 33:r.addTask(a[s-1],a[s]),this.$=`task`;break;case 34:this.$=a[s-1],r.setClickEvent(a[s-1],a[s],null);break;case 35:this.$=a[s-2],r.setClickEvent(a[s-2],a[s-1],a[s]);break;case 36:this.$=a[s-2],r.setClickEvent(a[s-2],a[s-1],null),r.setLink(a[s-2],a[s]);break;case 37:this.$=a[s-3],r.setClickEvent(a[s-3],a[s-2],a[s-1]),r.setLink(a[s-3],a[s]);break;case 38:this.$=a[s-2],r.setClickEvent(a[s-2],a[s],null),r.setLink(a[s-2],a[s-1]);break;case 39:this.$=a[s-3],r.setClickEvent(a[s-3],a[s-1],a[s]),r.setLink(a[s-3],a[s-2]);break;case 40:this.$=a[s-1],r.setLink(a[s-1],a[s]);break;case 41:case 47:this.$=a[s-1]+` `+a[s];break;case 42:case 43:case 45:this.$=a[s-2]+` `+a[s-1]+` `+a[s];break;case 44:case 46:this.$=a[s-3]+` `+a[s-2]+` `+a[s-1]+` `+a[s]}},`anonymous`),table:[{3:1,4:[1,2]},{1:[3]},e(t,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:n,13:r,14:i,15:a,16:o,17:s,18:c,19:18,20:l,21:u,22:d,23:f,24:p,25:m,26:h,27:g,28:_,29:v,30:y,31:b,33:x,35:S,36:C,37:24,38:T,40:E},e(t,[2,7],{1:[2,1]}),e(t,[2,3]),{9:36,11:17,12:n,13:r,14:i,15:a,16:o,17:s,18:c,19:18,20:l,21:u,22:d,23:f,24:p,25:m,26:h,27:g,28:_,29:v,30:y,31:b,33:x,35:S,36:C,37:24,38:T,40:E},e(t,[2,5]),e(t,[2,6]),e(t,[2,17]),e(t,[2,18]),e(t,[2,19]),e(t,[2,20]),e(t,[2,21]),e(t,[2,22]),e(t,[2,23]),e(t,[2,24]),e(t,[2,25]),e(t,[2,26]),e(t,[2,27]),{32:[1,37]},{34:[1,38]},e(t,[2,30]),e(t,[2,31]),e(t,[2,32]),{39:[1,39]},e(t,[2,8]),e(t,[2,9]),e(t,[2,10]),e(t,[2,11]),e(t,[2,12]),e(t,[2,13]),e(t,[2,14]),e(t,[2,15]),e(t,[2,16]),{41:[1,40],43:[1,41]},e(t,[2,4]),e(t,[2,28]),e(t,[2,29]),e(t,[2,33]),e(t,[2,34],{42:[1,42],43:[1,43]}),e(t,[2,40],{41:[1,44]}),e(t,[2,35],{43:[1,45]}),e(t,[2,36]),e(t,[2,38],{42:[1,46]}),e(t,[2,37]),e(t,[2,39])],defaultActions:{},parseError:w(function(e,t){if(t.recoverable)this.trace(e);else{var n=Error(e);throw n.hash=t,n}},`parseError`),parse:w(function(e){var t=this,n=[0],r=[],i=[null],a=[],o=this.table,s=``,c=0,l=0,u=0,d=2,f=1,p=a.slice.call(arguments,1),m=Object.create(this.lexer),h={yy:{}};for(var g in this.yy)Object.prototype.hasOwnProperty.call(this.yy,g)&&(h.yy[g]=this.yy[g]);m.setInput(e,h.yy),h.yy.lexer=m,h.yy.parser=this,m.yylloc===void 0&&(m.yylloc={});var _=m.yylloc;a.push(_);var v=m.options&&m.options.ranges;this.parseError=typeof h.yy.parseError==`function`?h.yy.parseError:Object.getPrototypeOf(this).parseError;function y(e){n.length-=2*e,i.length-=e,a.length-=e}w(y,`popStack`);function b(){var e=r.pop()||m.lex()||f;return typeof e!=`number`&&(e instanceof Array&&(r=e,e=r.pop()),e=t.symbols_[e]||e),e}w(b,`lex`);for(var x,S,C,T,E,D={},O,k,A,j;;){if(C=n[n.length-1],this.defaultActions[C]?T=this.defaultActions[C]:(x??=b(),T=o[C]&&o[C][x]),T===void 0||!T.length||!T[0]){var ee=``;for(O in j=[],o[C])this.terminals_[O]&&O>d&&j.push(`'`+this.terminals_[O]+`'`);ee=m.showPosition?`Parse error on line `+(c+1)+`:
`+m.showPosition()+`
Expecting `+j.join(`, `)+`, got '`+(this.terminals_[x]||x)+`'`:`Parse error on line `+(c+1)+`: Unexpected `+(x==f?`end of input`:`'`+(this.terminals_[x]||x)+`'`),this.parseError(ee,{text:m.match,token:this.terminals_[x]||x,line:m.yylineno,loc:_,expected:j})}if(T[0]instanceof Array&&T.length>1)throw Error(`Parse Error: multiple actions possible at state: `+C+`, token: `+x);switch(T[0]){case 1:n.push(x),i.push(m.yytext),a.push(m.yylloc),n.push(T[1]),x=null,S?(x=S,S=null):(l=m.yyleng,s=m.yytext,c=m.yylineno,_=m.yylloc,u>0&&u--);break;case 2:if(k=this.productions_[T[1]][1],D.$=i[i.length-k],D._$={first_line:a[a.length-(k||1)].first_line,last_line:a[a.length-1].last_line,first_column:a[a.length-(k||1)].first_column,last_column:a[a.length-1].last_column},v&&(D._$.range=[a[a.length-(k||1)].range[0],a[a.length-1].range[1]]),E=this.performAction.apply(D,[s,l,c,h.yy,T[1],i,a].concat(p)),E!==void 0)return E;k&&(n=n.slice(0,-1*k*2),i=i.slice(0,-1*k),a=a.slice(0,-1*k)),n.push(this.productions_[T[1]][0]),i.push(D.$),a.push(D._$),A=o[n[n.length-2]][n[n.length-1]],n.push(A);break;case 3:return!0}}return!0},`parse`)};D.lexer=(function(){return{EOF:1,parseError:w(function(e,t){if(this.yy.parser)this.yy.parser.parseError(e,t);else throw Error(e)},`parseError`),setInput:w(function(e,t){return this.yy=t||this.yy||{},this._input=e,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match=``,this.conditionStack=[`INITIAL`],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},`setInput`),input:w(function(){var e=this._input[0];return this.yytext+=e,this.yyleng++,this.offset++,this.match+=e,this.matched+=e,e.match(/(?:\r\n?|\n).*/g)?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),e},`input`),unput:w(function(e){var t=e.length,n=e.split(/(?:\r\n?|\n)/g);this._input=e+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-t),this.offset-=t;var r=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),n.length-1&&(this.yylineno-=n.length-1);var i=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:n?(n.length===r.length?this.yylloc.first_column:0)+r[r.length-n.length].length-n[0].length:this.yylloc.first_column-t},this.options.ranges&&(this.yylloc.range=[i[0],i[0]+this.yyleng-t]),this.yyleng=this.yytext.length,this},`unput`),more:w(function(){return this._more=!0,this},`more`),reject:w(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError(`Lexical error on line `+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:``,token:null,line:this.yylineno});return this},`reject`),less:w(function(e){this.unput(this.match.slice(e))},`less`),pastInput:w(function(){var e=this.matched.substr(0,this.matched.length-this.match.length);return(e.length>20?`...`:``)+e.substr(-20).replace(/\n/g,``)},`pastInput`),upcomingInput:w(function(){var e=this.match;return e.length<20&&(e+=this._input.substr(0,20-e.length)),(e.substr(0,20)+(e.length>20?`...`:``)).replace(/\n/g,``)},`upcomingInput`),showPosition:w(function(){var e=this.pastInput(),t=Array(e.length+1).join(`-`);return e+this.upcomingInput()+`
`+t+`^`},`showPosition`),test_match:w(function(e,t){var n,r,i;if(this.options.backtrack_lexer&&(i={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(i.yylloc.range=this.yylloc.range.slice(0))),r=e[0].match(/(?:\r\n?|\n).*/g),r&&(this.yylineno+=r.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:r?r[r.length-1].length-r[r.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+e[0].length},this.yytext+=e[0],this.match+=e[0],this.matches=e,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(e[0].length),this.matched+=e[0],n=this.performAction.call(this,this.yy,this,t,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),n)return n;if(this._backtrack){for(var a in i)this[a]=i[a];return!1}return!1},`test_match`),next:w(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var e,t,n,r;this._more||(this.yytext=``,this.match=``);for(var i=this._currentRules(),a=0;a<i.length;a++)if(n=this._input.match(this.rules[i[a]]),n&&(!t||n[0].length>t[0].length)){if(t=n,r=a,this.options.backtrack_lexer){if(e=this.test_match(n,i[a]),e!==!1)return e;if(this._backtrack){t=!1;continue}return!1}if(!this.options.flex)break}return t?(e=this.test_match(t,i[r]),e!==!1&&e):this._input===``?this.EOF:this.parseError(`Lexical error on line `+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:``,token:null,line:this.yylineno})},`next`),lex:w(function(){return this.next()||this.lex()},`lex`),begin:w(function(e){this.conditionStack.push(e)},`begin`),popState:w(function(){return this.conditionStack.length-1>0?this.conditionStack.pop():this.conditionStack[0]},`popState`),_currentRules:w(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},`_currentRules`),topState:w(function(e){return e=this.conditionStack.length-1-Math.abs(e||0),e>=0?this.conditionStack[e]:`INITIAL`},`topState`),pushState:w(function(e){this.begin(e)},`pushState`),stateStackSize:w(function(){return this.conditionStack.length},`stateStackSize`),options:{"case-insensitive":!0},performAction:w(function(e,t,n,r){switch(n){case 0:return this.begin(`open_directive`),`open_directive`;case 1:return this.begin(`acc_title`),31;case 2:return this.popState(),`acc_title_value`;case 3:return this.begin(`acc_descr`),33;case 4:return this.popState(),`acc_descr_value`;case 5:this.begin(`acc_descr_multiline`);break;case 6:this.popState();break;case 7:return`acc_descr_multiline_value`;case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin(`href`);break;case 15:this.popState();break;case 16:return 43;case 17:this.begin(`callbackname`);break;case 18:this.popState();break;case 19:this.popState(),this.begin(`callbackargs`);break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin(`click`);break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return`date`;case 45:return 30;case 46:return`accDescription`;case 47:return 36;case 48:return 38;case 49:return 39;case 50:return`:`;case 51:return 6;case 52:return`INVALID`}},`anonymous`),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}}})();function O(){this.yy={}}return w(O,`Parser`),O.prototype=D,D.Parser=O,new O})();ue.parser=ue;var de=ue;R.default.extend(oe.default),R.default.extend(se.default),R.default.extend(ce.default);var fe={friday:5,saturday:6},z=``,pe=``,B=void 0,V=``,H=[],U=[],me=new Map,he=[],W=[],G=``,ge=``,_e=[`active`,`done`,`crit`,`milestone`,`vert`],ve=[],K=``,q=!1,ye=!1,be=`sunday`,J=`saturday`,xe=0,Se=w(function(){he=[],W=[],G=``,ve=[],nt=0,at=void 0,X=void 0,Z=[],z=``,pe=``,ge=``,B=void 0,V=``,H=[],U=[],q=!1,ye=!1,xe=0,me=new Map,K=``,A(),be=`sunday`,J=`saturday`},`clear`),Ce=w(function(e){K=e},`setDiagramId`),we=w(function(e){pe=e},`setAxisFormat`),Te=w(function(){return pe},`getAxisFormat`),Ee=w(function(e){B=e},`setTickInterval`),De=w(function(){return B},`getTickInterval`),Oe=w(function(e){V=e},`setTodayMarker`),ke=w(function(){return V},`getTodayMarker`),Ae=w(function(e){z=e},`setDateFormat`),je=w(function(){q=!0},`enableInclusiveEndDates`),Me=w(function(){return q},`endDatesAreInclusive`),Ne=w(function(){ye=!0},`enableTopAxis`),Pe=w(function(){return ye},`topAxisEnabled`),Fe=w(function(e){ge=e},`setDisplayMode`),Ie=w(function(){return ge},`getDisplayMode`),Le=w(function(){return z},`getDateFormat`),Re=w((e,t)=>{let n=t.toLowerCase().split(/[\s,]+/).filter(e=>e!==``);return[...new Set([...e,...n])]},`mergeTokens`),ze=w(function(e){H=Re(H,e)},`setIncludes`),Be=w(function(){return H},`getIncludes`),Ve=w(function(e){U=Re(U,e)},`setExcludes`),He=w(function(){return U},`getExcludes`),Ue=w(function(){return me},`getLinks`),We=w(function(e){G=e,he.push(e)},`addSection`),Ge=w(function(){return he},`getSections`),Ke=w(function(){let e=lt(),t=0;for(;!e&&t<10;)e=lt(),t++;return W=Z,W},`getTasks`),qe=w(function(e,t,n,r){let i=e.format(t.trim()),a=e.format(`YYYY-MM-DD`);return r.includes(i)||r.includes(a)?!1:n.includes(`weekends`)&&(e.isoWeekday()===fe[J]||e.isoWeekday()===fe[J]+1)||n.includes(e.format(`dddd`).toLowerCase())?!0:n.includes(i)||n.includes(a)},`isInvalidDate`),Je=w(function(e){be=e},`setWeekday`),Ye=w(function(){return be},`getWeekday`),Xe=w(function(e){J=e},`setWeekend`),Ze=w(function(e,t,n,r){if(!n.length||e.manualEndTime)return;let i;i=e.startTime instanceof Date?(0,R.default)(e.startTime):(0,R.default)(e.startTime,t,!0),i=i.add(1,`d`);let a;a=e.endTime instanceof Date?(0,R.default)(e.endTime):(0,R.default)(e.endTime,t,!0);let[o,s]=Qe(i,a,t,n,r);e.endTime=o.toDate(),e.renderEndTime=s},`checkTaskDates`),Qe=w(function(e,t,n,r,i){let a=!1,o=null,s=t.add(1e4,`d`);for(;e<=t;){if(a||(o=t.toDate()),a=qe(e,n,r,i),a&&(t=t.add(1,`d`),t>s))throw Error("Failed to find a valid date that was not excluded by `excludes` after 10,000 iterations.");e=e.add(1,`d`)}return[t,o]},`fixTaskDates`),$e=w(function(e,t,n){if(n=n.trim(),w(e=>{let t=e.trim();return t===`x`||t===`X`},`isTimestampFormat`)(t)&&/^\d+$/.test(n))return new Date(Number(n));let r=/^after\s+(?<ids>[\d\w- ]+)/.exec(n);if(r!==null){let e=null;for(let t of r.groups.ids.split(` `)){let n=Q(t);n!==void 0&&(!e||n.endTime>e.endTime)&&(e=n)}if(e)return e.endTime;let t=new Date;return t.setHours(0,0,0,0),t}let i=(0,R.default)(n,t.trim(),!0);if(i.isValid())return i.toDate();{E.debug(`Invalid date:`+n),E.debug(`With date format:`+t.trim());let e=new Date(n);if(e===void 0||isNaN(e.getTime())||e.getFullYear()<-1e4||e.getFullYear()>1e4)throw Error(`Invalid date:`+n);return e}},`getStartDate`),et=w(function(e){let t=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(e.trim());return t===null?[NaN,`ms`]:[Number.parseFloat(t[1]),t[2]]},`parseDuration`),tt=w(function(e,t,n,r=!1){n=n.trim();let i=/^until\s+(?<ids>[\d\w- ]+)/.exec(n);if(i!==null){let e=null;for(let t of i.groups.ids.split(` `)){let n=Q(t);n!==void 0&&(!e||n.startTime<e.startTime)&&(e=n)}if(e)return e.startTime;let t=new Date;return t.setHours(0,0,0,0),t}let a=(0,R.default)(n,t.trim(),!0);if(a.isValid())return r&&(a=a.add(1,`d`)),a.toDate();let o=(0,R.default)(e),[s,c]=et(n);if(!Number.isNaN(s)){let e=o.add(s,c);e.isValid()&&(o=e)}return o.toDate()},`getEndDate`),nt=0,Y=w(function(e){return e===void 0?(nt+=1,`task`+nt):e},`parseId`),rt=w(function(e,t){let n;n=t.substr(0,1)===`:`?t.substr(1,t.length):t;let r=n.split(`,`),i={};ht(r,i,_e);for(let e=0;e<r.length;e++)r[e]=r[e].trim();let a=``;switch(r.length){case 1:i.id=Y(),i.startTime=e.endTime,a=r[0];break;case 2:i.id=Y(),i.startTime=$e(void 0,z,r[0]),a=r[1];break;case 3:i.id=Y(r[0]),i.startTime=$e(void 0,z,r[1]),a=r[2]}return a&&(i.endTime=tt(i.startTime,z,a,q),i.manualEndTime=(0,R.default)(a,`YYYY-MM-DD`,!0).isValid(),Ze(i,z,U,H)),i},`compileData`),it=w(function(e,t){let n;n=t.substr(0,1)===`:`?t.substr(1,t.length):t;let r=n.split(`,`),i={};ht(r,i,_e);for(let e=0;e<r.length;e++)r[e]=r[e].trim();switch(r.length){case 1:i.id=Y(),i.startTime={type:`prevTaskEnd`,id:e},i.endTime={data:r[0]};break;case 2:i.id=Y(),i.startTime={type:`getStartDate`,startData:r[0]},i.endTime={data:r[1]};break;case 3:i.id=Y(r[0]),i.startTime={type:`getStartDate`,startData:r[1]},i.endTime={data:r[2]}}return i},`parseData`),at,X,Z=[],ot={},st=w(function(e,t){let n={section:G,type:G,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:t},task:e,classes:[]},r=it(X,t);n.raw.startTime=r.startTime,n.raw.endTime=r.endTime,n.id=r.id,n.prevTaskId=X,n.active=r.active,n.done=r.done,n.crit=r.crit,n.milestone=r.milestone,n.vert=r.vert,n.vert?n.order=-1:(n.order=xe,xe++);let i=Z.push(n);X=n.id,ot[n.id]=i-1},`addTask`),Q=w(function(e){let t=ot[e];return Z[t]},`findTaskById`),ct=w(function(e,t){let n={section:G,type:G,description:e,task:e,classes:[]},r=rt(at,t);n.startTime=r.startTime,n.endTime=r.endTime,n.id=r.id,n.active=r.active,n.done=r.done,n.crit=r.crit,n.milestone=r.milestone,n.vert=r.vert,at=n,W.push(n)},`addTaskOrg`),lt=w(function(){let e=w(function(e){let t=Z[e],n=``;switch(Z[e].raw.startTime.type){case`prevTaskEnd`:t.startTime=Q(t.prevTaskId).endTime;break;case`getStartDate`:n=$e(void 0,z,Z[e].raw.startTime.startData),n&&(Z[e].startTime=n)}return Z[e].startTime&&(Z[e].endTime=tt(Z[e].startTime,z,Z[e].raw.endTime.data,q),Z[e].endTime&&(Z[e].processed=!0,Z[e].manualEndTime=(0,R.default)(Z[e].raw.endTime.data,`YYYY-MM-DD`,!0).isValid(),Ze(Z[e],z,U,H))),Z[e].processed},`compileTask`),t=!0;for(let[n,r]of Z.entries())e(n),t&&=r.processed;return t},`compileTasks`),ut=w(function(e,t){let n=t;P().securityLevel!==`loose`&&(n=(0,ae.sanitizeUrl)(t)),e.split(`,`).forEach(function(e){Q(e)!==void 0&&(pt(e,()=>{window.open(n,`_self`)}),me.set(e,n))}),dt(e,`clickable`)},`setLink`),dt=w(function(e,t){e.split(`,`).forEach(function(e){let n=Q(e);n!==void 0&&n.classes.push(t)})},`setClass`),ft=w(function(e,t,n){if(P().securityLevel!==`loose`||t===void 0)return;let r=[];if(typeof n==`string`){r=n.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let e=0;e<r.length;e++){let t=r[e].trim();t.startsWith(`"`)&&t.endsWith(`"`)&&(t=t.substr(1,t.length-2)),r[e]=t}}r.length===0&&r.push(e),Q(e)!==void 0&&pt(e,()=>{ne.runFunc(t,...r)})},`setClickFun`),pt=w(function(e,t){ve.push(function(){let n=K?`${K}-${e}`:e,r=document.querySelector(`[id="${n}"]`);r!==null&&r.addEventListener(`click`,function(){t()})},function(){let n=K?`${K}-${e}`:e,r=document.querySelector(`[id="${n}-text"]`);r!==null&&r.addEventListener(`click`,function(){t()})})},`pushFun`),mt={getConfig:w(()=>P().gantt,`getConfig`),clear:Se,setDateFormat:Ae,getDateFormat:Le,enableInclusiveEndDates:je,endDatesAreInclusive:Me,enableTopAxis:Ne,topAxisEnabled:Pe,setAxisFormat:we,getAxisFormat:Te,setTickInterval:Ee,getTickInterval:De,setTodayMarker:Oe,getTodayMarker:ke,setAccTitle:k,getAccTitle:F,setDiagramTitle:O,getDiagramTitle:N,setDiagramId:Ce,setDisplayMode:Fe,getDisplayMode:Ie,setAccDescription:D,getAccDescription:M,addSection:We,getSections:Ge,getTasks:Ke,addTask:st,findTaskById:Q,addTaskOrg:ct,setIncludes:ze,getIncludes:Be,setExcludes:Ve,getExcludes:He,setClickEvent:w(function(e,t,n){e.split(`,`).forEach(function(e){ft(e,t,n)}),dt(e,`clickable`)},`setClickEvent`),setLink:ut,getLinks:Ue,bindFunctions:w(function(e){ve.forEach(function(t){t(e)})},`bindFunctions`),parseDuration:et,isInvalidDate:qe,setWeekday:Je,getWeekday:Ye,setWeekend:Xe};function ht(e,t,n){let r=!0;for(;r;)r=!1,n.forEach(function(n){let i=`^\\s*`+n+`\\s*$`,a=new RegExp(i);e[0].match(a)&&(t[n]=!0,e.shift(1),r=!0)})}w(ht,`getTaskTags`),R.default.extend(le.default);var gt=w(function(){E.debug(`Something is calling, setConf, remove the call`)},`setConf`),_t={monday:a,tuesday:p,wednesday:x,thursday:s,friday:f,saturday:g,sunday:y},vt=w((e,t)=>{let n=[...e].map(()=>-1/0),r=[...e].sort((e,t)=>e.startTime-t.startTime||e.order-t.order),i=0;for(let e of r)for(let r=0;r<n.length;r++)if(e.startTime>=n[r]){n[r]=e.endTime,e.order=r+t,r>i&&(i=r);break}return i},`getMaxIntersections`),$,yt=1e4,bt={parser:de,db:mt,renderer:{setConf:gt,draw:w(function(t,n,a,s){let f=P().gantt;s.db.setDiagramId(n);let p=P().securityLevel,g;p===`sandbox`&&(g=e(`#i`+n));let y=e(p===`sandbox`?g.nodes()[0].contentDocument.body:`body`),x=p===`sandbox`?g.nodes()[0].contentDocument:document,T=x.getElementById(n);$=T.parentElement.offsetWidth,$===void 0&&($=1200),f.useWidth!==void 0&&($=f.useWidth);let D=s.db.getTasks(),O=D.filter(e=>!e.vert),k=[];for(let e of O)k.push(e.type);k=se(k);let A={},M=2*f.topPadding;if(s.db.getDisplayMode()===`compact`||f.displayMode===`compact`){let e={};for(let t of O)e[t.section]===void 0?e[t.section]=[t]:e[t.section].push(t);let t=0;for(let n of Object.keys(e)){let r=vt(e[n],t)+1;t+=r,M+=r*(f.barHeight+f.barGap),A[n]=r}}else{M+=O.length*(f.barHeight+f.barGap);for(let e of k)A[e]=O.filter(t=>t.type===e).length}T.setAttribute(`viewBox`,`0 0 `+$+` `+M);let N=y.select(`[id="${n}"]`),F=b().domain([C(D,function(e){return e.startTime}),o(D,function(e){return e.endTime})]).rangeRound([0,$-f.leftPadding-f.rightPadding]);function te(e,t){let n=e.startTime,r=t.startTime,i=0;return n>r?i=1:n<r&&(i=-1),i}w(te,`taskCompare`),D.sort(te),ne(D,$,M),j(N,M,$,f.useMaxWidth),N.append(`text`).text(s.db.getDiagramTitle()).attr(`x`,$/2).attr(`y`,f.titleTopMargin).attr(`class`,`titleText`);function ne(e,t,n){let i=f.barHeight,a=i+f.barGap,o=f.topPadding,c=f.leftPadding,l=r().domain([0,k.length]).range([`#00B9FA`,`#F95002`]).interpolate(u);ie(a,o,c,t,n,e,s.db.getExcludes(),s.db.getIncludes()),L(c,o,t,n),re(e,a,o,c,i,l,t,n),ae(a,o,c,i,l),oe(c,o,t,n)}w(ne,`makeGantt`);function re(t,r,i,a,o,c,l){t.sort((e,t)=>e.vert===t.vert?0:e.vert?1:-1);let u=t.filter(e=>!e.vert),d=[...new Set(u.map(e=>e.order))].map(e=>u.find(t=>t.order===e));N.append(`g`).selectAll(`rect`).data(d).enter().append(`rect`).attr(`x`,0).attr(`y`,function(e,t){return t=e.order,t*r+i-2}).attr(`width`,function(){return l-f.rightPadding/2}).attr(`height`,r).attr(`class`,function(e){for(let[t,n]of k.entries())if(e.type===n)return`section section`+t%f.numberSectionStyles;return`section section0`}).enter();let p=N.append(`g`).selectAll(`rect`).data(t).enter(),m=s.db.getLinks();if(p.append(`rect`).attr(`id`,function(e){return n+`-`+e.id}).attr(`rx`,3).attr(`ry`,3).attr(`x`,function(e){return e.milestone?F(e.startTime)+a+.5*(F(e.endTime)-F(e.startTime))-.5*o:F(e.startTime)+a}).attr(`y`,function(e,t){return t=e.order,e.vert?f.gridLineStartPadding:t*r+i}).attr(`width`,function(e){return e.milestone?o:e.vert?.08*o:F(e.renderEndTime||e.endTime)-F(e.startTime)}).attr(`height`,function(e){return e.vert?u.length*(f.barHeight+f.barGap)+f.barHeight*2:o}).attr(`transform-origin`,function(e,t){return t=e.order,(F(e.startTime)+a+.5*(F(e.endTime)-F(e.startTime))).toString()+`px `+(t*r+i+.5*o).toString()+`px`}).attr(`class`,function(e){let t=``;e.classes.length>0&&(t=e.classes.join(` `));let n=0;for(let[t,r]of k.entries())e.type===r&&(n=t%f.numberSectionStyles);let r=``;return e.active?e.crit?r+=` activeCrit`:r=` active`:e.done?r=e.crit?` doneCrit`:` done`:e.crit&&(r+=` crit`),r.length===0&&(r=` task`),e.milestone&&(r=` milestone `+r),e.vert&&(r=` vert `+r),r+=n,r+=` `+t,`task`+r}),p.append(`text`).attr(`id`,function(e){return n+`-`+e.id+`-text`}).text(function(e){return e.task}).attr(`font-size`,f.fontSize).attr(`x`,function(e){let t=F(e.startTime),n=F(e.renderEndTime||e.endTime);if(e.milestone&&(t+=.5*(F(e.endTime)-F(e.startTime))-.5*o,n=t+o),e.vert)return F(e.startTime)+a;let r=this.getBBox().width;return r>n-t?n+r+1.5*f.leftPadding>l?t+a-5:n+a+5:(n-t)/2+t+a}).attr(`y`,function(e,t){return e.vert?f.gridLineStartPadding+u.length*(f.barHeight+f.barGap)+60:(t=e.order,t*r+f.barHeight/2+(f.fontSize/2-2)+i)}).attr(`text-height`,o).attr(`class`,function(e){let t=F(e.startTime),n=F(e.endTime);e.milestone&&(n=t+o);let r=this.getBBox().width,i=``;e.classes.length>0&&(i=e.classes.join(` `));let a=0;for(let[t,n]of k.entries())e.type===n&&(a=t%f.numberSectionStyles);let s=``;return e.active&&(s=e.crit?`activeCritText`+a:`activeText`+a),e.done?s=e.crit?s+` doneCritText`+a:s+` doneText`+a:e.crit&&(s=s+` critText`+a),e.milestone&&(s+=` milestoneText`),e.vert&&(s+=` vertText`),r>n-t?n+r+1.5*f.leftPadding>l?i+` taskTextOutsideLeft taskTextOutside`+a+` `+s:i+` taskTextOutsideRight taskTextOutside`+a+` `+s+` width-`+r:i+` taskText taskText`+a+` `+s+` width-`+r}),P().securityLevel===`sandbox`){let t;t=e(`#i`+n);let r=t.nodes()[0].contentDocument;p.filter(function(e){return m.has(e.id)}).each(function(e){var t=r.querySelector(`#`+CSS.escape(n+`-`+e.id)),i=r.querySelector(`#`+CSS.escape(n+`-`+e.id+`-text`));let a=t.parentNode;var o=r.createElement(`a`);o.setAttribute(`xlink:href`,m.get(e.id)),o.setAttribute(`target`,`_top`),a.appendChild(o),o.appendChild(t),o.appendChild(i)})}}w(re,`drawRects`);function ie(e,t,r,i,a,o,c,l){if(c.length===0&&l.length===0)return;let u,d;for(let{startTime:e,endTime:t}of o)(u===void 0||e<u)&&(u=e),(d===void 0||t>d)&&(d=t);if(!u||!d)return;if((0,R.default)(d).diff((0,R.default)(u),`year`)>5){E.warn(`The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.`);return}let p=s.db.getDateFormat(),m=[],h=null,g=(0,R.default)(u);for(;g.valueOf()<=d;)s.db.isInvalidDate(g,p,c,l)?h?h.end=g:h={start:g,end:g}:h&&=(m.push(h),null),g=g.add(1,`d`);N.append(`g`).selectAll(`rect`).data(m).enter().append(`rect`).attr(`id`,e=>n+`-exclude-`+e.start.format(`YYYY-MM-DD`)).attr(`x`,e=>F(e.start.startOf(`day`))+r).attr(`y`,f.gridLineStartPadding).attr(`width`,e=>F(e.end.endOf(`day`))-F(e.start.startOf(`day`))).attr(`height`,a-t-f.gridLineStartPadding).attr(`transform-origin`,function(t,n){return(F(t.start)+r+.5*(F(t.end)-F(t.start))).toString()+`px `+(n*e+.5*a).toString()+`px`}).attr(`class`,`exclude-range`)}w(ie,`drawExcludeDays`);function I(e,t,n,r){if(n<=0||e>t)return 1/0;let i=t-e,a=R.default.duration({[r??`day`]:n}).asMilliseconds();return a<=0?1/0:Math.ceil(i/a)}w(I,`getEstimatedTickCount`);function L(e,t,n,r){let a=s.db.getDateFormat(),o=s.db.getAxisFormat(),u;u=o||(a===`D`?`%d`:f.axisFormat??`%Y-%m-%d`);let p=i(F).tickSize(-r+t+f.gridLineStartPadding).tickFormat(h(u)),g=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(s.db.getTickInterval()||f.tickInterval);if(g!==null){let e=parseInt(g[1],10);if(isNaN(e)||e<=0)E.warn(`Invalid tick interval value: "${g[1]}". Skipping custom tick interval.`);else{let t=g[2],n=s.db.getWeekday()||f.weekday,r=F.domain(),i=r[0],a=r[1],o=I(i,a,e,t);if(o>yt)E.warn(`The tick interval "${e}${t}" would generate ${o} ticks, which exceeds the maximum allowed (${yt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(t){case`millisecond`:p.ticks(d.every(e));break;case`second`:p.ticks(m.every(e));break;case`minute`:p.ticks(_.every(e));break;case`hour`:p.ticks(l.every(e));break;case`day`:p.ticks(c.every(e));break;case`week`:p.ticks(_t[n].every(e));break;case`month`:p.ticks(v.every(e))}}}if(N.append(`g`).attr(`class`,`grid`).attr(`transform`,`translate(`+e+`, `+(r-50)+`)`).call(p).selectAll(`text`).style(`text-anchor`,`middle`).attr(`fill`,`#000`).attr(`stroke`,`none`).attr(`font-size`,10).attr(`dy`,`1em`),s.db.topAxisEnabled()||f.topAxis){let n=S(F).tickSize(-r+t+f.gridLineStartPadding).tickFormat(h(u));if(g!==null){let e=parseInt(g[1],10);if(isNaN(e)||e<=0)E.warn(`Invalid tick interval value: "${g[1]}". Skipping custom tick interval.`);else{let t=g[2],r=s.db.getWeekday()||f.weekday,i=F.domain(),a=i[0],o=i[1];if(I(a,o,e,t)<=yt)switch(t){case`millisecond`:n.ticks(d.every(e));break;case`second`:n.ticks(m.every(e));break;case`minute`:n.ticks(_.every(e));break;case`hour`:n.ticks(l.every(e));break;case`day`:n.ticks(c.every(e));break;case`week`:n.ticks(_t[r].every(e));break;case`month`:n.ticks(v.every(e))}}}N.append(`g`).attr(`class`,`grid`).attr(`transform`,`translate(`+e+`, `+t+`)`).call(n).selectAll(`text`).style(`text-anchor`,`middle`).attr(`fill`,`#000`).attr(`stroke`,`none`).attr(`font-size`,10)}}w(L,`makeGrid`);function ae(e,t){let n=0,r=Object.keys(A).map(e=>[e,A[e]]);N.append(`g`).selectAll(`text`).data(r).enter().append(function(e){let t=e[0].split(ee.lineBreakRegex),n=-(t.length-1)/2,r=x.createElementNS(`http://www.w3.org/2000/svg`,`text`);r.setAttribute(`dy`,n+`em`);for(let[e,n]of t.entries()){let t=x.createElementNS(`http://www.w3.org/2000/svg`,`tspan`);t.setAttribute(`alignment-baseline`,`central`),t.setAttribute(`x`,`10`),e>0&&t.setAttribute(`dy`,`1em`),t.textContent=n,r.appendChild(t)}return r}).attr(`x`,10).attr(`y`,function(i,a){if(a>0)for(let o=0;o<a;o++)return n+=r[a-1][1],i[1]*e/2+n*e+t;else return i[1]*e/2+t}).attr(`font-size`,f.sectionFontSize).attr(`class`,function(e){for(let[t,n]of k.entries())if(e[0]===n)return`sectionTitle sectionTitle`+t%f.numberSectionStyles;return`sectionTitle`})}w(ae,`vertLabels`);function oe(e,t,n,r){let i=s.db.getTodayMarker();if(i===`off`)return;let a=N.append(`g`).attr(`class`,`today`),o=new Date,c=a.append(`line`);c.attr(`x1`,F(o)+e).attr(`x2`,F(o)+e).attr(`y1`,f.titleTopMargin).attr(`y2`,r-f.titleTopMargin).attr(`class`,`today`),i!==``&&c.attr(`style`,i.replace(/,/g,`;`))}w(oe,`drawToday`);function se(e){let t={},n=[];for(let r=0,i=e.length;r<i;++r)Object.prototype.hasOwnProperty.call(t,e[r])||(t[e[r]]=!0,n.push(e[r]));return n}w(se,`checkUnique`)},`draw`)},styles:w(e=>`
  .mermaid-main-font {
        font-family: ${e.fontFamily};
  }

  .exclude-range {
    fill: ${e.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${e.sectionBkgColor};
  }

  .section2 {
    fill: ${e.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${e.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${e.titleColor};
  }

  .sectionTitle1 {
    fill: ${e.titleColor};
  }

  .sectionTitle2 {
    fill: ${e.titleColor};
  }

  .sectionTitle3 {
    fill: ${e.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${e.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${e.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${e.fontFamily};
    fill: ${e.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${e.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${e.taskTextDarkColor};
    text-anchor: start;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${e.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${e.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${e.taskBkgColor};
    stroke: ${e.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${e.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${e.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${e.activeTaskBkgColor};
    stroke: ${e.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${e.doneTaskBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  /* Done task text displayed outside the bar sits against the diagram background,
     not against the done-task bar, so it must use the outside/contrast color. */
  .doneText0.taskTextOutsideLeft,
  .doneText0.taskTextOutsideRight,
  .doneText1.taskTextOutsideLeft,
  .doneText1.taskTextOutsideRight,
  .doneText2.taskTextOutsideLeft,
  .doneText2.taskTextOutsideRight,
  .doneText3.taskTextOutsideLeft,
  .doneText3.taskTextOutsideRight {
    fill: ${e.taskTextOutsideColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  /* Done-crit task text outside the bar \u2014 same reasoning as doneText above. */
  .doneCritText0.taskTextOutsideLeft,
  .doneCritText0.taskTextOutsideRight,
  .doneCritText1.taskTextOutsideLeft,
  .doneCritText1.taskTextOutsideRight,
  .doneCritText2.taskTextOutsideLeft,
  .doneCritText2.taskTextOutsideRight,
  .doneCritText3.taskTextOutsideLeft,
  .doneCritText3.taskTextOutsideRight {
    fill: ${e.taskTextOutsideColor} !important;
  }

  .vert {
    stroke: ${e.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${e.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${e.titleColor||e.textColor};
    font-family: ${e.fontFamily};
  }
`,`getStyles`)};export{bt as diagram};