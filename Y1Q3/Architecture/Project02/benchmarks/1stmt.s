gcc2_compiled.:
___gnu_compiled_c:
.text
	.align 4
	.proc	0110
_label_rtx:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,40
	be L2
	nop
	call _abort,0
	nop
L2:
	ld [%fp+68],%o0
	ld [%o0+64],%o1
	cmp %o1,0
	be L3
	nop
	ld [%fp+68],%o0
	ld [%o0+64],%i0
	b L1
	nop
L3:
	call _gen_label_rtx,0
	nop
	ld [%fp+68],%o1
	st %o0,[%o1+64]
	mov %o0,%i0
	b L1
	nop
L1:
	ret
	restore
	.align 4
	.global _emit_jump
	.proc	020
_emit_jump:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp+68],%o0
	call _gen_jump,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	call _emit_barrier,0
	nop
L4:
	ret
	restore
	.align 4
	.global _expand_label
	.proc	020
_expand_label:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp+68],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_label,0
	nop
	sethi %hi(_stack_block_stack),%o0
	ld [%o0+%lo(_stack_block_stack)],%o1
	cmp %o1,0
	be L6
	nop
	mov 8,%o0
	call _oballoc,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(_stack_block_stack),%o2
	ld [%o2+%lo(_stack_block_stack)],%o1
	ld [%o1+36],%o2
	st %o2,[%o0]
	sethi %hi(_stack_block_stack),%o1
	ld [%o1+%lo(_stack_block_stack)],%o0
	ld [%fp-20],%o1
	st %o1,[%o0+36]
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	st %o1,[%o0+4]
L6:
L5:
	ret
	restore
	.align 4
	.global _expand_goto
	.proc	020
_expand_goto:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	call _expand_goto_internal,0
	nop
L7:
	ret
	restore
	.align 8
LC0:
	.ascii "jump to `%s' invalidly jumps into binding contour\0"
	.align 4
	.proc	020
_expand_goto_internal:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %g0,[%fp-24]
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,17
	be L9
	nop
	call _abort,0
	nop
L9:
	ld [%fp+72],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L10
	nop
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
L11:
	ld [%fp-20],%o0
	cmp %o0,0
	be L12
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	ld [%fp+72],%o0
	ld [%o1+4],%o1
	ld [%o0+4],%o0
	cmp %o1,%o0
	bge L14
	nop
	b L12
	nop
L14:
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	be L15
	nop
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	st %o1,[%fp-24]
L15:
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L16
	nop
	ld [%fp-20],%o1
	ld [%o1+28],%o0
	mov 0,%o1
	call _expand_cleanups,0
	nop
L16:
L13:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L11
	nop
L12:
	ld [%fp-24],%o0
	cmp %o0,0
	be L17
	nop
	sethi %hi(_stack_pointer_rtx),%o1
	ld [%o1+%lo(_stack_pointer_rtx)],%o0
	ld [%fp-24],%o1
	call _emit_move_insn,0
	nop
L17:
	ld [%fp+68],%o0
	cmp %o0,0
	be L18
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	sethi %hi(524288),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	be L18
	nop
	ld [%fp+68],%o0
	ld [%o0+36],%o1
	sethi %hi(LC0),%o2
	or %o2,%lo(LC0),%o0
	ld [%o1+20],%o1
	call _error,0
	nop
L18:
	b L19
	nop
L10:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%fp+76],%o2
	call _expand_fixup,0
	nop
	cmp %o0,0
	bne L20
	nop
	ld [%fp+68],%o0
	cmp %o0,0
	be L21
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	sethi %hi(16384),%o2
	or %o1,%o2,%o1
	st %o1,[%o0+12]
L21:
L20:
L19:
	ld [%fp+72],%o0
	call _emit_jump,0
	nop
L8:
	ret
	restore
	.align 4
	.proc	04
_expand_fixup:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	cmp %o1,0
	be L23
	nop
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o0
	ld [%fp+72],%o1
	ld [%o0+16],%o0
	cmp %o1,%o0
	be L24
	nop
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o0
	ld [%fp+72],%o1
	ld [%o0+20],%o0
	cmp %o1,%o0
	be L24
	nop
	b L23
	nop
L24:
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	st %o1,[%fp-24]
	b L25
	nop
L23:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	cmp %o1,0
	be L26
	nop
	sethi %hi(_loop_stack),%o1
	ld [%o1+%lo(_loop_stack)],%o0
	ld [%fp+72],%o1
	ld [%o0+16],%o0
	cmp %o1,%o0
	be L27
	nop
	sethi %hi(_loop_stack),%o1
	ld [%o1+%lo(_loop_stack)],%o0
	ld [%fp+72],%o1
	ld [%o0+20],%o0
	cmp %o1,%o0
	be L27
	nop
	sethi %hi(_loop_stack),%o1
	ld [%o1+%lo(_loop_stack)],%o0
	ld [%fp+72],%o1
	ld [%o0+24],%o0
	cmp %o1,%o0
	be L27
	nop
	b L26
	nop
L27:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	st %o1,[%fp-24]
	b L28
	nop
L26:
	st %g0,[%fp-24]
L28:
L25:
	ld [%fp-24],%o0
	cmp %o0,0
	be L29
	nop
	ld [%fp-24],%o0
	ld [%o0],%o1
	st %o1,[%fp-28]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
L30:
	ld [%fp-28],%o0
	cmp %o0,0
	be L31
	nop
	ld [%fp-28],%o0
	ld [%fp-20],%o1
	cmp %o0,%o1
	be L31
	nop
	ld [%fp-28],%o0
	ld [%o0],%o1
	st %o1,[%fp-28]
	b L30
	nop
L31:
	ld [%fp-28],%o0
	cmp %o0,0
	be L32
	nop
	mov 0,%i0
	b L22
	nop
L32:
	sethi %hi(_block_stack),%o1
	ld [%o1+%lo(_block_stack)],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
L33:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	be L34
	nop
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	cmp %o0,%o1
	bne L36
	nop
	ld [%fp-28],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
L36:
L35:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L33
	nop
L34:
	ld [%fp-28],%o0
	st %o0,[%fp-24]
L29:
	nop
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
L37:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	be L38
	nop
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	bne L41
	nop
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L41
	nop
	b L40
	nop
L41:
	b L38
	nop
L40:
L39:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L37
	nop
L38:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	be L42
	nop
	mov 24,%o0
	call _oballoc,0
	nop
	st %o0,[%fp-28]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-28],%l0
	ld [%fp+76],%l1
	ld [%fp+76],%o0
	cmp %o0,0
	bne L43
	nop
	call _get_last_insn,0
	nop
	mov %o0,%l1
L43:
	st %l1,[%l0+4]
	ld [%fp-28],%o0
	ld [%fp+68],%o1
	st %o1,[%o0+8]
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	st %o1,[%o0+12]
	ld [%fp-28],%o0
	st %g0,[%o0+16]
	ld [%fp-28],%l0
	ld [%fp-20],%o0
	ld [%o0+32],%o1
	cmp %o1,0
	bne L46
	nop
	ld [%fp-20],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	bne L46
	nop
	b L44
	nop
L46:
	ld [%fp-20],%o1
	ld [%fp-20],%o2
	mov 0,%o0
	ld [%o1+28],%o1
	ld [%o2+32],%o2
	call _tree_cons,0
	nop
	b L45
	nop
L44:
	mov 0,%o0
L45:
	st %o0,[%l0+20]
	ld [%fp-28],%o0
	sethi %hi(_goto_fixup_chain),%o1
	ld [%o1+%lo(_goto_fixup_chain)],%o2
	st %o2,[%o0]
	sethi %hi(_goto_fixup_chain),%o0
	ld [%fp-28],%o1
	st %o1,[%o0+%lo(_goto_fixup_chain)]
L42:
	ld [%fp-20],%o0
	xor %o0,0,%o1
	subcc %g0,%o1,%g0
	addx %g0,0,%o0
	mov %o0,%i0
	b L22
	nop
L22:
	ret
	restore
	.align 8
LC1:
	.ascii "label `%s' used before containing binding contour\0"
	.align 4
	.proc	020
_fixup_gotos:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	mov 0,%l1
	sethi %hi(_goto_fixup_chain),%o0
	ld [%o0+%lo(_goto_fixup_chain)],%l0
L48:
	cmp %l0,0
	be L49
	nop
	ld [%l0+4],%o0
	cmp %o0,0
	bne L51
	nop
	cmp %l1,0
	be L52
	nop
	ld [%l0],%o0
	st %o0,[%l1]
L52:
	b L53
	nop
L51:
	ld [%l0+12],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	be L54
	nop
	ld [%l0+8],%o0
	cmp %o0,0
	be L55
	nop
	ld [%fp+84],%o0
	cmp %o0,0
	bne L56
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	bne L56
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	bne L56
	nop
	b L55
	nop
L56:
	ld [%fp+80],%o0
	ld [%l0+4],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	cmp %o0,%o1
	ble L55
	nop
	ld [%l0+8],%o0
	ld [%o0+12],%o1
	sethi %hi(16384),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	bne L55
	nop
	ld [%l0+8],%o0
	sethi %hi(LC1),%o2
	or %o2,%lo(LC1),%o1
	call _error_with_decl,0
	nop
	ld [%l0+8],%o0
	ld [%o0+12],%o1
	sethi %hi(16384),%o2
	or %o1,%o2,%o1
	st %o1,[%o0+12]
L55:
	ld [%l0+20],%o0
	cmp %o0,0
	be L57
	nop
	ld [%l0+20],%o0
	st %o0,[%fp-20]
L58:
	ld [%fp-20],%o0
	cmp %o0,0
	be L59
	nop
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	sethi %hi(16384),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	be L61
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	cmp %o1,0
	be L61
	nop
	ld [%fp-20],%o0
	add %l0,4,%o1
	ld [%o0+20],%o0
	call _fixup_cleanups,0
	nop
L61:
L60:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L58
	nop
L59:
L57:
	ld [%l0+16],%o0
	cmp %o0,0
	be L62
	nop
	sethi %hi(_stack_pointer_rtx),%o1
	ld [%o1+%lo(_stack_pointer_rtx)],%o0
	ld [%l0+16],%o1
	call _gen_move_insn,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%l0+4],%o1
	call _emit_insn_after,0
	nop
L62:
	st %g0,[%l0+4]
	b L63
	nop
L54:
	ld [%fp+68],%o0
	cmp %o0,0
	be L64
	nop
	ld [%l0+20],%o0
	st %o0,[%fp-20]
L65:
	ld [%fp-20],%o0
	cmp %o0,0
	be L66
	nop
	ld [%fp-20],%o0
	ld [%fp+68],%o1
	ld [%o0+4],%o0
	ld [%o1+32],%o1
	cmp %o0,%o1
	bne L68
	nop
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	sethi %hi(16384),%o2
	or %o1,%o2,%o1
	st %o1,[%o0+12]
L68:
L67:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L65
	nop
L66:
	ld [%fp+72],%o0
	cmp %o0,0
	be L69
	nop
	ld [%fp+72],%o0
	st %o0,[%l0+16]
L69:
L64:
L63:
L53:
L50:
	mov %l0,%l1
	ld [%l0],%l0
	b L48
	nop
L49:
L47:
	ret
	restore
	.align 4
	.global _expand_asm
	.proc	020
_expand_asm:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o2
	mov 21,%o0
	mov 0,%o1
	ld [%o2+24],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L70:
	ret
	restore
	.align 8
LC2:
	.ascii "input operand constraint contains `+'\0"
	.align 8
LC3:
	.ascii "output operand constraint lacks `='\0"
	.align 8
LC4:
	.ascii "more than %d operands in `asm'\0"
	.align 8
LC5:
	.ascii "\0"
	.align 8
LC6:
	.ascii "hard register `%s' listed as input operand to `asm'\0"
	.align 8
LC7:
	.ascii "input operand constraint contains `%c'\0"
	.align 8
LC8:
	.ascii "unknown register name `%s' in `asm'\0"
	.align 4
	.global _expand_asm_operands
	.proc	020
_expand_asm_operands:
	!#PROLOGUE# 0
	save %sp,-192,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	st %i4,[%fp+84]
	st %i5,[%fp+88]
	ld [%fp+76],%o0
	call _list_length,0
	nop
	st %o0,[%fp-32]
	ld [%fp+72],%o0
	call _list_length,0
	nop
	st %o0,[%fp-36]
	ld [%fp+80],%o0
	call _list_length,0
	nop
	st %o0,[%fp-40]
	ld [%fp-36],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	add %o2,7,%o0
	add %sp,108,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,108,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-48]
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	mov 0,%l0
	ld [%fp+72],%o0
	st %o0,[%fp-44]
L72:
	ld [%fp-44],%o0
	cmp %o0,0
	be L73
	nop
	ld [%fp-44],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-56]
	ld [%fp-56],%o0
	sethi %hi(_error_mark_node),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_error_mark_node)],%o1
	cmp %o0,%o1
	bne L75
	nop
	b L71
	nop
L75:
	st %g0,[%fp-64]
	st %g0,[%fp-60]
L76:
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%fp-60],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bge L77
	nop
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%o1+24],%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,43
	bne L79
	nop
	sethi %hi(LC2),%o1
	or %o1,%lo(LC2),%o0
	call _error,0
	nop
	b L71
	nop
L79:
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%o1+24],%o0
	ld [%fp-60],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,61
	bne L80
	nop
	mov 1,%o0
	st %o0,[%fp-64]
L80:
L78:
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L76
	nop
L77:
	ld [%fp-64],%o0
	cmp %o0,0
	bne L81
	nop
	sethi %hi(LC3),%o1
	or %o1,%lo(LC3),%o0
	call _error,0
	nop
	b L71
	nop
L81:
	ld [%fp-56],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	be L82
	nop
	ld [%fp-56],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	be L82
	nop
	ld [%fp-56],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,49
	be L82
	nop
	ld [%fp-56],%o0
	ld [%o0+8],%o1
	ldub [%o1+28],%o0
	and %o0,0xff,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%fp-68]
	mov 112,%o0
	ld [%fp-56],%o1
	ld [%fp-68],%o2
	call _build_nt,0
	nop
	st %o0,[%fp-72]
	sethi %hi(_save_expr_regs),%o3
	mov 2,%o0
	mov 0,%o1
	ld [%fp-68],%o2
	ld [%o3+%lo(_save_expr_regs)],%o3
	call _gen_rtx,0
	nop
	sethi %hi(_save_expr_regs),%o1
	st %o0,[%o1+%lo(_save_expr_regs)]
	ld [%fp-44],%o0
	ld [%fp-72],%o1
	st %o1,[%o0+20]
	ld [%fp-72],%o0
	ld [%fp-56],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+8]
L82:
	ld [%fp-44],%o1
	ld [%o1+20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %l0,%o1
	sll %o1,2,%o2
	ld [%fp-48],%o3
	add %o2,%o3,%o1
	st %o0,[%o1]
L74:
	ld [%fp-44],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-44]
	add %l0,1,%l0
	b L72
	nop
L73:
	ld [%fp-32],%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	cmp %o0,5
	ble L83
	nop
	sethi %hi(LC4),%o1
	or %o1,%lo(LC4),%o0
	mov 5,%o1
	call _error,0
	nop
	b L71
	nop
L83:
	ld [%fp-32],%o0
	call _rtvec_alloc,0
	nop
	st %o0,[%fp-20]
	ld [%fp-32],%o0
	call _rtvec_alloc,0
	nop
	st %o0,[%fp-24]
	ld [%fp+68],%o2
	ld [%fp-24],%o0
	st %o0,[%sp+92]
	ld [%fp+88],%o0
	st %o0,[%sp+96]
	ld [%fp+92],%o0
	st %o0,[%sp+100]
	mov 22,%o0
	mov 0,%o1
	ld [%o2+24],%o2
	sethi %hi(LC5),%o4
	or %o4,%lo(LC5),%o3
	mov 0,%o4
	ld [%fp-20],%o5
	call _gen_rtx,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%fp+84],%o1
	and %o1,1,%o2
	sll %o2,4,%o1
	ld [%o0],%o3
	and %o3,-17,%o2
	or %o2,%o1,%o2
	st %o2,[%o0]
	mov 0,%l0
	ld [%fp+76],%o0
	st %o0,[%fp-44]
L84:
	ld [%fp-44],%o0
	cmp %o0,0
	be L85
	nop
	ld [%fp-44],%o0
	ld [%o0+20],%o1
	sethi %hi(_error_mark_node),%o0
	ld [%o1+8],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	bne L87
	nop
	b L71
	nop
L87:
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	bne L88
	nop
	ld [%fp-44],%o0
	ld [%o0+20],%o1
	sethi %hi(LC6),%o2
	or %o2,%lo(LC6),%o0
	ld [%o1+24],%o1
	call _error,0
	nop
	b L71
	nop
L88:
	nop
	st %g0,[%fp-72]
L89:
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%fp-72],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bge L90
	nop
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%o1+24],%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,61
	be L93
	nop
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%o1+24],%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o2
	sra %o2,24,%o0
	cmp %o0,43
	be L93
	nop
	b L92
	nop
L93:
	ld [%fp-44],%o0
	ld [%o0+16],%o1
	ld [%o1+24],%o0
	ld [%fp-72],%o1
	add %o0,%o1,%o0
	ldub [%o0],%o1
	sll %o1,24,%o0
	sra %o0,24,%o1
	sethi %hi(LC7),%o2
	or %o2,%lo(LC7),%o0
	call _error,0
	nop
	b L71
	nop
L92:
L91:
	ld [%fp-72],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-72]
	b L89
	nop
L90:
	ld [%fp-44],%o1
	ld [%o1+20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	ld [%fp-28],%o2
	ld [%o2+16],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
	ld [%fp-44],%o0
	ld [%o0+20],%o1
	ld [%o1+8],%o0
	ldub [%o0+28],%o2
	and %o2,0xff,%o1
	ld [%fp-44],%o0
	ld [%o0+16],%o2
	mov 21,%o0
	ld [%o2+24],%o2
	call _gen_rtx,0
	nop
	ld [%fp-28],%o2
	ld [%o2+20],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
	add %l0,1,%l0
L86:
	ld [%fp-44],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-44]
	b L84
	nop
L85:
	nop
	mov 0,%l0
L94:
	ld [%fp-32],%o0
	cmp %l0,%o0
	bge L95
	nop
	ld [%fp-28],%o1
	ld [%o1+16],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov 0,%o1
	call _protect_from_queue,0
	nop
	ld [%fp-28],%o2
	ld [%o2+16],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L96:
	add %l0,1,%l0
	b L94
	nop
L95:
	nop
	mov 0,%l0
L97:
	ld [%fp-36],%o0
	cmp %l0,%o0
	bge L98
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp-48],%o0
	add %o1,%o0,%o1
	ld [%o1],%o0
	mov 1,%o1
	call _protect_from_queue,0
	nop
	mov %l0,%o1
	sll %o1,2,%o2
	ld [%fp-48],%o3
	add %o2,%o3,%o1
	st %o0,[%o1]
L99:
	add %l0,1,%l0
	b L97
	nop
L98:
	ld [%fp-36],%o0
	cmp %o0,1
	bne L100
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	bne L100
	nop
	ld [%fp-28],%o0
	ld [%fp+72],%o1
	ld [%o1+16],%o2
	ld [%o2+24],%o1
	st %o1,[%o0+8]
	ld [%fp-48],%o2
	mov 25,%o0
	mov 0,%o1
	ld [%o2],%o2
	ld [%fp-28],%o3
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
	st %o0,[%fp-52]
	b L101
	nop
L100:
	ld [%fp-36],%o0
	cmp %o0,0
	bne L102
	nop
	ld [%fp-40],%o0
	cmp %o0,0
	bne L102
	nop
	ld [%fp-28],%o0
	call _emit_insn,0
	nop
	st %o0,[%fp-52]
	b L103
	nop
L102:
	ld [%fp-28],%o0
	st %o0,[%fp-72]
	ld [%fp-36],%o0
	st %o0,[%fp-64]
	ld [%fp-64],%o0
	cmp %o0,0
	bne L104
	nop
	mov 1,%o0
	st %o0,[%fp-64]
L104:
	ld [%fp-64],%o0
	ld [%fp-40],%o2
	add %o0,%o2,%o1
	mov %o1,%o0
	call _rtvec_alloc,0
	nop
	mov %o0,%o2
	mov 20,%o0
	mov 0,%o1
	call _gen_rtx,0
	nop
	st %o0,[%fp-28]
	mov 0,%l0
	ld [%fp+72],%o0
	st %o0,[%fp-44]
L105:
	ld [%fp-44],%o0
	cmp %o0,0
	be L106
	nop
	mov %l0,%o0
	sll %o0,2,%o1
	ld [%fp-48],%o0
	add %o1,%o0,%l1
	ld [%fp+68],%o2
	ld [%fp-44],%o0
	ld [%o0+16],%o3
	ld [%fp-24],%o0
	st %o0,[%sp+92]
	ld [%fp+88],%o0
	st %o0,[%sp+96]
	ld [%fp+92],%o0
	st %o0,[%sp+100]
	mov 22,%o0
	mov 0,%o1
	ld [%o2+24],%o2
	ld [%o3+24],%o3
	mov %l0,%o4
	ld [%fp-20],%o5
	call _gen_rtx,0
	nop
	mov %o0,%o3
	mov 25,%o0
	mov 0,%o1
	ld [%l1],%o2
	call _gen_rtx,0
	nop
	ld [%fp-28],%o2
	ld [%o2+4],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
	ld [%fp-28],%o1
	ld [%o1+4],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%o0+4],%o1
	ld [%o1+8],%o0
	ld [%fp+84],%o1
	and %o1,1,%o2
	sll %o2,4,%o1
	ld [%o0],%o3
	and %o3,-17,%o2
	or %o2,%o1,%o2
	st %o2,[%o0]
L107:
	ld [%fp-44],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-44]
	add %l0,1,%l0
	b L105
	nop
L106:
	cmp %l0,0
	bne L108
	nop
	ld [%fp-28],%o1
	ld [%o1+4],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o0
	ld [%fp-72],%o1
	st %o1,[%o0+4]
	add %l0,1,%l0
L108:
	nop
	ld [%fp+80],%o0
	st %o0,[%fp-44]
L109:
	ld [%fp-44],%o0
	cmp %o0,0
	be L110
	nop
	ld [%fp-44],%o0
	ld [%o0+20],%o1
	ld [%o1+24],%o0
	st %o0,[%fp-76]
	st %g0,[%fp-60]
L112:
	ld [%fp-60],%o0
	cmp %o0,55
	bg L113
	nop
	ld [%fp-60],%o0
	mov %o0,%o2
	sll %o2,2,%o1
	sethi %hi(_reg_names),%o0
	or %o0,%lo(_reg_names),%o2
	ld [%fp-76],%o0
	ld [%o1+%o2],%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L115
	nop
	b L113
	nop
L115:
L114:
	ld [%fp-60],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-60]
	b L112
	nop
L113:
	ld [%fp-60],%o0
	cmp %o0,56
	bne L116
	nop
	sethi %hi(LC8),%o1
	or %o1,%lo(LC8),%o0
	ld [%fp-76],%o1
	call _error,0
	nop
	b L71
	nop
L116:
	mov 34,%o0
	mov 1,%o1
	ld [%fp-60],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o2
	mov 27,%o0
	mov 0,%o1
	call _gen_rtx,0
	nop
	ld [%fp-28],%o2
	ld [%o2+4],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L111:
	ld [%fp-44],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-44]
	add %l0,1,%l0
	b L109
	nop
L110:
	ld [%fp-28],%o0
	call _emit_insn,0
	nop
	st %o0,[%fp-52]
L103:
L101:
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L71:
	ret
	restore
	.align 8
LC9:
	.ascii "statement with no effect\0"
	.align 4
	.global _expand_expr_stmt
	.proc	020
_expand_expr_stmt:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_extra_warnings),%o0
	ld [%o0+%lo(_extra_warnings)],%o1
	cmp %o1,0
	be L118
	nop
	sethi %hi(_expr_stmts_for_value),%o0
	ld [%o0+%lo(_expr_stmts_for_value)],%o1
	cmp %o1,0
	bne L118
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	sethi %hi(1048576),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	bne L118
	nop
	sethi %hi(_error_mark_node),%o0
	ld [%fp+68],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	be L118
	nop
	sethi %hi(_emit_filename),%o0
	sethi %hi(_emit_lineno),%o1
	ld [%o0+%lo(_emit_filename)],%o0
	ld [%o1+%lo(_emit_lineno)],%o1
	sethi %hi(LC9),%o3
	or %o3,%lo(LC9),%o2
	call _warning_with_file_and_line,0
	nop
L118:
	sethi %hi(_last_expr_type),%o0
	ld [%fp+68],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_last_expr_type)]
	sethi %hi(_flag_syntax_only),%o0
	ld [%o0+%lo(_flag_syntax_only)],%o1
	cmp %o1,0
	bne L119
	nop
	sethi %hi(_expr_stmts_for_value),%o0
	ld [%o0+%lo(_expr_stmts_for_value)],%o1
	cmp %o1,0
	bne L120
	nop
	sethi %hi(_const0_rtx),%o0
	ld [%o0+%lo(_const0_rtx)],%o1
	b L121
	nop
L120:
	mov 0,%o1
L121:
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	sethi %hi(_last_expr_value),%o1
	st %o0,[%o1+%lo(_last_expr_value)]
L119:
	call _emit_queue,0
	nop
L117:
	ret
	restore
	.align 4
	.global _clear_last_expr
	.proc	020
_clear_last_expr:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L122:
	ret
	restore
	.align 4
	.global _expand_start_stmt_expr
	.proc	0111
_expand_start_stmt_expr:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	call _start_sequence,0
	nop
	st %o0,[%fp-20]
	call _suspend_momentary,0
	nop
	st %o0,[%fp-24]
	mov 113,%o0
	call _make_node,0
	nop
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	call _resume_momentary,0
	nop
	mov 24,%o0
	ld [%fp-28],%o1
	add %o0,%o1,%o0
	ld [%fp-20],%o1
	st %o1,[%o0]
	sethi %hi(_expr_stmts_for_value),%o1
	sethi %hi(_expr_stmts_for_value),%o0
	sethi %hi(_expr_stmts_for_value),%o1
	ld [%o1+%lo(_expr_stmts_for_value)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_expr_stmts_for_value)]
	ld [%fp-28],%i0
	b L123
	nop
L123:
	ret
	restore
	.align 4
	.global _expand_end_stmt_expr
	.proc	0111
_expand_end_stmt_expr:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov 24,%o0
	ld [%fp+68],%o1
	add %o0,%o1,%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	call _do_pending_stack_adjust,0
	nop
	sethi %hi(_last_expr_type),%o0
	ld [%o0+%lo(_last_expr_type)],%o1
	cmp %o1,0
	bne L125
	nop
	sethi %hi(_last_expr_type),%o0
	sethi %hi(_void_type_node),%o1
	ld [%o1+%lo(_void_type_node)],%o2
	st %o2,[%o0+%lo(_last_expr_type)]
	sethi %hi(_last_expr_value),%o0
	sethi %hi(_const0_rtx),%o1
	ld [%o1+%lo(_const0_rtx)],%o2
	st %o2,[%o0+%lo(_last_expr_value)]
L125:
	ld [%fp+68],%o0
	sethi %hi(_last_expr_type),%o1
	ld [%o1+%lo(_last_expr_type)],%o2
	st %o2,[%o0+8]
	mov 24,%o0
	ld [%fp+68],%o1
	add %o0,%o1,%o0
	sethi %hi(_last_expr_value),%o1
	ld [%o1+%lo(_last_expr_value)],%o2
	st %o2,[%o0]
	call _get_insns,0
	nop
	mov 20,%o1
	ld [%fp+68],%o2
	add %o1,%o2,%o1
	st %o0,[%o1]
	sethi %hi(_rtl_expr_chain),%o2
	mov 0,%o0
	ld [%fp+68],%o1
	ld [%o2+%lo(_rtl_expr_chain)],%o2
	call _tree_cons,0
	nop
	sethi %hi(_rtl_expr_chain),%o1
	st %o0,[%o1+%lo(_rtl_expr_chain)]
	ld [%fp-20],%o0
	call _end_sequence,0
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	sethi %hi(1048576),%o2
	or %o1,%o2,%o1
	st %o1,[%o0+12]
	sethi %hi(_last_expr_value),%o1
	ld [%o1+%lo(_last_expr_value)],%o0
	call _volatile_refs_p,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	and %o1,1,%o2
	sll %o2,12,%o1
	ld [%o0+12],%o2
	sethi %hi(-4097),%o4
	or %o4,%lo(-4097),%o3
	and %o2,%o3,%o2
	or %o2,%o1,%o2
	st %o2,[%o0+12]
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_expr_stmts_for_value),%o1
	sethi %hi(_expr_stmts_for_value),%o0
	sethi %hi(_expr_stmts_for_value),%o1
	ld [%o1+%lo(_expr_stmts_for_value)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_expr_stmts_for_value)]
	ld [%fp+68],%i0
	b L124
	nop
L124:
	ret
	restore
	.align 4
	.global _expand_start_cond
	.proc	020
_expand_start_cond:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	mov 40,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o2
	st %o2,[%o0+4]
	ld [%fp-20],%o0
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o2
	st %o2,[%o0]
	ld [%fp-20],%o0
	sethi %hi(_nesting_depth),%o2
	sethi %hi(_nesting_depth),%o1
	sethi %hi(_nesting_depth),%o2
	ld [%o2+%lo(_nesting_depth)],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%o1+%lo(_nesting_depth)]
	st %o3,[%o0+8]
	ld [%fp-20],%o0
	st %g0,[%o0+20]
	call _gen_label_rtx,0
	nop
	ld [%fp-20],%o1
	st %o0,[%o1+16]
	ld [%fp-20],%o0
	ld [%fp+72],%o1
	cmp %o1,0
	be L127
	nop
	ld [%fp-20],%o2
	ld [%o2+16],%o1
	b L128
	nop
L127:
	mov 0,%o1
L128:
	st %o1,[%o0+12]
	sethi %hi(_cond_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_cond_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_nesting_stack)]
	ld [%fp-20],%o1
	ld [%fp+68],%o0
	ld [%o1+16],%o1
	mov 0,%o2
	call _do_jump,0
	nop
L126:
	ret
	restore
	.align 4
	.global _expand_end_cond
	.proc	020
_expand_end_cond:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	st %o1,[%fp-20]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-20],%o1
	ld [%o1+16],%o0
	call _emit_label,0
	nop
L130:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L133:
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	st %o1,[%fp-28]
	sethi %hi(_cond_stack),%o0
	ld [%fp-28],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_cond_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-28],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-28],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-28],%o0
	call _free,0
	nop
L135:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-24],%o0
	cmp %o1,%o0
	ble L134
	nop
	b L133
	nop
L134:
L132:
	b L131
	nop
	b L130
	nop
L131:
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L129:
	ret
	restore
	.align 4
	.global _expand_start_else
	.proc	020
_expand_start_else:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	call _gen_label_rtx,0
	nop
	sethi %hi(_cond_stack),%o2
	ld [%o2+%lo(_cond_stack)],%o1
	st %o0,[%o1+20]
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o0
	ld [%o0+12],%o1
	cmp %o1,0
	be L137
	nop
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o0
	sethi %hi(_cond_stack),%o2
	ld [%o2+%lo(_cond_stack)],%o1
	ld [%o1+20],%o2
	st %o2,[%o0+12]
L137:
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	ld [%o1+20],%o0
	call _emit_jump,0
	nop
	sethi %hi(_cond_stack),%o1
	ld [%o1+%lo(_cond_stack)],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	be L138
	nop
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	ld [%o1+16],%o0
	call _emit_label,0
	nop
L138:
L136:
	ret
	restore
	.align 4
	.global _expand_end_else
	.proc	020
_expand_end_else:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	st %o1,[%fp-20]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	cmp %o1,0
	be L140
	nop
	ld [%fp-20],%o1
	ld [%o1+20],%o0
	call _emit_label,0
	nop
L140:
	nop
L141:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L144:
	sethi %hi(_cond_stack),%o0
	ld [%o0+%lo(_cond_stack)],%o1
	st %o1,[%fp-28]
	sethi %hi(_cond_stack),%o0
	ld [%fp-28],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_cond_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-28],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-28],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-28],%o0
	call _free,0
	nop
L146:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-24],%o0
	cmp %o1,%o0
	ble L145
	nop
	b L144
	nop
L145:
L143:
	b L142
	nop
	b L141
	nop
L142:
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L139:
	ret
	restore
	.align 4
	.global _expand_start_loop
	.proc	020
_expand_start_loop:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov 40,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	st %o1,[%l0+4]
	sethi %hi(_nesting_stack),%o0
	ld [%o0+%lo(_nesting_stack)],%o1
	st %o1,[%l0]
	sethi %hi(_nesting_depth),%o1
	sethi %hi(_nesting_depth),%o0
	sethi %hi(_nesting_depth),%o1
	ld [%o1+%lo(_nesting_depth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	st %o2,[%l0+8]
	call _gen_label_rtx,0
	nop
	st %o0,[%l0+16]
	call _gen_label_rtx,0
	nop
	st %o0,[%l0+20]
	ld [%l0+16],%o0
	st %o0,[%l0+24]
	ld [%fp+68],%o0
	cmp %o0,0
	be L148
	nop
	ld [%l0+20],%o0
	b L149
	nop
L148:
	mov 0,%o0
L149:
	st %o0,[%l0+12]
	sethi %hi(_loop_stack),%o0
	st %l0,[%o0+%lo(_loop_stack)]
	sethi %hi(_nesting_stack),%o0
	st %l0,[%o0+%lo(_nesting_stack)]
	call _do_pending_stack_adjust,0
	nop
	call _emit_queue,0
	nop
	mov 0,%o0
	mov -4,%o1
	call _emit_note,0
	nop
	ld [%l0+16],%o0
	call _emit_label,0
	nop
L147:
	ret
	restore
	.align 4
	.global _expand_start_loop_continue_elsewhere
	.proc	020
_expand_start_loop_continue_elsewhere:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	call _expand_start_loop,0
	nop
	call _gen_label_rtx,0
	nop
	sethi %hi(_loop_stack),%o2
	ld [%o2+%lo(_loop_stack)],%o1
	st %o0,[%o1+24]
L150:
	ret
	restore
	.align 4
	.global _expand_loop_continue_here
	.proc	020
_expand_loop_continue_here:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	call _do_pending_stack_adjust,0
	nop
	mov 0,%o0
	mov -8,%o1
	call _emit_note,0
	nop
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	ld [%o1+24],%o0
	call _emit_label,0
	nop
L151:
	ret
	restore
	.align 4
	.global _expand_end_loop
	.proc	020
_expand_end_loop:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	call _get_last_insn,0
	nop
	mov %o0,%l0
	sethi %hi(_loop_stack),%o1
	ld [%o1+%lo(_loop_stack)],%o0
	ld [%o0+16],%l1
	call _do_pending_stack_adjust,0
	nop
	sethi %hi(_optimize),%o0
	ld [%o0+%lo(_optimize)],%o1
	cmp %o1,0
	be L153
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L154
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,25
	bne L154
	nop
	ld [%l0+16],%o0
	sethi %hi(_pc_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_pc_rtx)],%o1
	cmp %o0,%o1
	bne L154
	nop
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,42
	bne L154
	nop
	b L153
	nop
L154:
	sethi %hi(_loop_stack),%o1
	ld [%o1+%lo(_loop_stack)],%o0
	ld [%o0+16],%l0
L155:
	cmp %l0,0
	be L156
	nop
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,14
	bne L158
	nop
	ld [%l0+16],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,25
	bne L158
	nop
	ld [%l0+16],%o0
	sethi %hi(_pc_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_pc_rtx)],%o1
	cmp %o0,%o1
	bne L158
	nop
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,42
	bne L158
	nop
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	ld [%o1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,38
	bne L160
	nop
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	ld [%o1+8],%o0
	sethi %hi(_loop_stack),%o2
	ld [%o2+%lo(_loop_stack)],%o1
	ld [%o0+4],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	be L159
	nop
	b L160
	nop
L160:
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	ld [%o1+12],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,38
	bne L158
	nop
	ld [%l0+16],%o0
	ld [%o0+8],%o1
	ld [%o1+12],%o0
	sethi %hi(_loop_stack),%o2
	ld [%o2+%lo(_loop_stack)],%o1
	ld [%o0+4],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	be L159
	nop
	b L158
	nop
L159:
	b L156
	nop
L158:
L157:
	ld [%l0+12],%l0
	b L155
	nop
L156:
	cmp %l0,0
	be L161
	nop
	call _gen_label_rtx,0
	nop
	mov %o0,%l2
	mov %l2,%o0
	ld [%l1+8],%o1
	call _emit_label_after,0
	nop
	call _get_last_insn,0
	nop
	mov %o0,%o2
	mov %l1,%o0
	mov %l0,%o1
	call _reorder_insns,0
	nop
	mov %l1,%o0
	call _gen_jump,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%l2+8],%o1
	call _emit_jump_insn_after,0
	nop
	ld [%l2+8],%o0
	call _emit_barrier_after,0
	nop
	mov %l2,%l1
L161:
L153:
	mov %l1,%o0
	call _emit_jump,0
	nop
	mov 0,%o0
	mov -5,%o1
	call _emit_note,0
	nop
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	ld [%o1+20],%o0
	call _emit_label,0
	nop
L162:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-20]
L165:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	st %o1,[%fp-24]
	sethi %hi(_loop_stack),%o0
	ld [%fp-24],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_loop_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-24],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-24],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-24],%o0
	call _free,0
	nop
L167:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-20],%o0
	cmp %o1,%o0
	ble L166
	nop
	b L165
	nop
L166:
L164:
	b L163
	nop
	b L162
	nop
L163:
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
L152:
	ret
	restore
	.align 4
	.global _expand_continue_loop
	.proc	04
_expand_continue_loop:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	cmp %o1,0
	bne L169
	nop
	mov 0,%i0
	b L168
	nop
L169:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	mov 0,%o0
	ld [%o1+24],%o1
	mov 0,%o2
	call _expand_goto_internal,0
	nop
	mov 1,%i0
	b L168
	nop
L168:
	ret
	restore
	.align 4
	.global _expand_exit_loop
	.proc	04
_expand_exit_loop:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	cmp %o1,0
	bne L171
	nop
	mov 0,%i0
	b L170
	nop
L171:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	mov 0,%o0
	ld [%o1+20],%o1
	mov 0,%o2
	call _expand_goto_internal,0
	nop
	mov 1,%i0
	b L170
	nop
L170:
	ret
	restore
	.align 4
	.global _expand_exit_loop_if_false
	.proc	04
_expand_exit_loop_if_false:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	cmp %o1,0
	bne L173
	nop
	mov 0,%i0
	b L172
	nop
L173:
	sethi %hi(_loop_stack),%o0
	ld [%o0+%lo(_loop_stack)],%o1
	ld [%fp+68],%o0
	ld [%o1+20],%o1
	mov 0,%o2
	call _do_jump,0
	nop
	mov 1,%i0
	b L172
	nop
L172:
	ret
	restore
	.align 4
	.global _expand_exit_something
	.proc	04
_expand_exit_something:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_nesting_stack),%o0
	ld [%o0+%lo(_nesting_stack)],%o1
	st %o1,[%fp-20]
L175:
	ld [%fp-20],%o0
	cmp %o0,0
	be L176
	nop
	ld [%fp-20],%o0
	ld [%o0+12],%o1
	cmp %o1,0
	be L178
	nop
	ld [%fp-20],%o1
	mov 0,%o0
	ld [%o1+12],%o1
	mov 0,%o2
	call _expand_goto_internal,0
	nop
	mov 1,%i0
	b L174
	nop
L178:
L177:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L175
	nop
L176:
	mov 0,%i0
	b L174
	nop
L174:
	ret
	restore
	.align 4
	.global _expand_null_return
	.proc	020
_expand_null_return:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov 0,%o0
	call _expand_null_return_1,0
	nop
L179:
	ret
	restore
	.align 4
	.proc	020
_expand_null_return_1:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	call _clear_pending_stack_adjust,0
	nop
	call _do_pending_stack_adjust,0
	nop
	sethi %hi(_last_expr_type),%o0
	st %g0,[%o0+%lo(_last_expr_type)]
	sethi %hi(_current_function_returns_pcc_struct),%o0
	ld [%o0+%lo(_current_function_returns_pcc_struct)],%o1
	cmp %o1,0
	be L181
	nop
	sethi %hi(_return_label),%o1
	mov 0,%o0
	ld [%o1+%lo(_return_label)],%o1
	ld [%fp+68],%o2
	call _expand_goto_internal,0
	nop
	b L180
	nop
L181:
	b L182
	nop
	call _gen_return,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	call _emit_barrier,0
	nop
	b L180
	nop
L182:
	sethi %hi(_return_label),%o1
	mov 0,%o0
	ld [%o1+%lo(_return_label)],%o1
	ld [%fp+68],%o2
	call _expand_goto_internal,0
	nop
L180:
	ret
	restore
	.align 4
	.global _expand_return
	.proc	020
_expand_return:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov 0,%l0
	st %g0,[%fp-24]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-28]
L184:
	ld [%fp-28],%o0
	cmp %o0,0
	be L185
	nop
	ld [%fp-28],%o0
	ld [%o0+28],%o1
	cmp %o1,0
	be L187
	nop
	mov 1,%o0
	st %o0,[%fp-24]
	b L185
	nop
L187:
L186:
	ld [%fp-28],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	b L184
	nop
L185:
	ld [%fp+68],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,45
	bne L188
	nop
	ld [%fp+68],%o0
	st %o0,[%fp-20]
	b L189
	nop
L188:
	ld [%fp+68],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,55
	be L191
	nop
	ld [%fp+68],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,56
	be L191
	nop
	b L190
	nop
L191:
	ld [%fp+68],%o0
	ld [%o0+20],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,45
	bne L190
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%o1
	st %o1,[%fp-20]
	b L192
	nop
L190:
	ld [%fp+68],%o0
	sethi %hi(_void_type_node),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_void_type_node)],%o1
	cmp %o0,%o1
	bne L193
	nop
	ld [%fp+68],%o0
	st %o0,[%fp-20]
	b L194
	nop
L193:
	st %g0,[%fp-20]
L194:
L192:
L189:
	sethi %hi(_optimize),%o0
	ld [%o0+%lo(_optimize)],%o1
	cmp %o1,0
	be L195
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L195
	nop
	sethi %hi(_frame_offset),%o0
	ld [%o0+%lo(_frame_offset)],%o1
	cmp %o1,0
	bne L195
	nop
	ld [%fp-20],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,60
	bne L195
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,114
	bne L195
	nop
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	sethi %hi(_this_function),%o0
	ld [%o1+20],%o1
	ld [%o0+%lo(_this_function)],%o0
	cmp %o1,%o0
	bne L195
	nop
	ld [%fp-20],%o0
	sethi %hi(_this_function),%o2
	ld [%o2+%lo(_this_function)],%o1
	ld [%o0+24],%o0
	ld [%o1+52],%o1
	call _tail_recursion_args,0
	nop
	cmp %o0,0
	be L195
	nop
	sethi %hi(_tail_recursion_label),%o0
	ld [%o0+%lo(_tail_recursion_label)],%o1
	cmp %o1,0
	bne L196
	nop
	call _gen_label_rtx,0
	nop
	sethi %hi(_tail_recursion_label),%o1
	st %o0,[%o1+%lo(_tail_recursion_label)]
	sethi %hi(_tail_recursion_label),%o0
	sethi %hi(_tail_recursion_reentry),%o1
	ld [%o0+%lo(_tail_recursion_label)],%o0
	ld [%o1+%lo(_tail_recursion_reentry)],%o1
	call _emit_label_after,0
	nop
L196:
	sethi %hi(_tail_recursion_label),%l2
	call _get_last_insn,0
	nop
	mov %o0,%o2
	mov 0,%o0
	ld [%l2+%lo(_tail_recursion_label)],%o1
	call _expand_goto_internal,0
	nop
	call _emit_barrier,0
	nop
	b L183
	nop
L195:
	b L197
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L198
	nop
	ld [%fp-20],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,105
	bgu L212
	nop
	cmp %o0,95
	blu L212
	nop
	b L206
	nop
L200:
L201:
L202:
L203:
L204:
L205:
L206:
L207:
L208:
L209:
L210:
	call _gen_label_rtx,0
	nop
	mov %o0,%l1
	sethi %hi(_this_function),%o1
	ld [%o1+%lo(_this_function)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%l0
	ld [%fp-20],%o0
	mov %l1,%o1
	call _jumpifnot,0
	nop
	sethi %hi(_const1_rtx),%o1
	mov %l0,%o0
	ld [%o1+%lo(_const1_rtx)],%o1
	call _emit_move_insn,0
	nop
	mov 26,%o0
	mov 0,%o1
	mov %l0,%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
	call _expand_null_return,0
	nop
	mov %l1,%o0
	call _emit_label,0
	nop
	sethi %hi(_const0_rtx),%o1
	mov %l0,%o0
	ld [%o1+%lo(_const0_rtx)],%o1
	call _emit_move_insn,0
	nop
	mov 26,%o0
	mov 0,%o1
	mov %l0,%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
	call _expand_null_return,0
	nop
	b L183
	nop
L212:
L199:
L198:
L197:
	ld [%fp-24],%o0
	cmp %o0,0
	be L213
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L213
	nop
	ld [%fp-20],%o0
	sethi %hi(_void_type_node),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_void_type_node)],%o1
	cmp %o0,%o1
	be L213
	nop
	sethi %hi(_this_function),%o1
	ld [%o1+%lo(_this_function)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L213
	nop
	ld [%fp-20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%l0
	call _emit_queue,0
	nop
	call _get_last_insn,0
	nop
	st %o0,[%fp-32]
	sethi %hi(_this_function),%o1
	ld [%o1+%lo(_this_function)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	mov %l0,%o1
	call _emit_move_insn,0
	nop
	sethi %hi(_this_function),%o1
	ld [%o1+%lo(_this_function)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%l0
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L214
	nop
	mov 26,%o0
	mov 0,%o1
	mov %l0,%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
L214:
	ld [%fp-32],%o0
	call _expand_null_return_1,0
	nop
	b L215
	nop
L213:
	ld [%fp+68],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%l0
	call _emit_queue,0
	nop
	sethi %hi(_this_function),%o1
	ld [%o1+%lo(_this_function)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%l0
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L216
	nop
	mov 26,%o0
	mov 0,%o1
	mov %l0,%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
L216:
	call _expand_null_return,0
	nop
L215:
L183:
	ret
	restore
	.align 4
	.global _drop_through_at_end_p
	.proc	04
_drop_through_at_end_p:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	call _get_last_insn,0
	nop
	st %o0,[%fp-20]
L218:
	ld [%fp-20],%o0
	cmp %o0,0
	be L219
	nop
	ld [%fp-20],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,18
	bne L219
	nop
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-20]
	b L218
	nop
L219:
	mov 0,%o0
	ld [%fp-20],%o1
	cmp %o1,0
	be L220
	nop
	ld [%fp-20],%o1
	lduh [%o1],%o2
	sll %o2,16,%o3
	srl %o3,16,%o1
	cmp %o1,16
	be L220
	nop
	mov 1,%o0
L220:
	mov %o0,%i0
	b L217
	nop
L217:
	ret
	restore
	.align 4
	.proc	04
_tail_recursion_args:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%l0
	ld [%fp+72],%l1
	ld [%fp+68],%l0
	ld [%fp+72],%l1
	mov 0,%l2
L222:
	cmp %l0,0
	be L223
	nop
	cmp %l1,0
	be L223
	nop
	ld [%l0+20],%o0
	ld [%o0+8],%o1
	ld [%l1+8],%o0
	cmp %o1,%o0
	be L225
	nop
	mov 0,%i0
	b L221
	nop
L225:
	ld [%l1+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L227
	nop
	ld [%l1+28],%o0
	cmp %o0,26
	be L227
	nop
	b L226
	nop
L227:
	mov 0,%i0
	b L221
	nop
L226:
L224:
	ld [%l0+4],%l0
	ld [%l1+4],%l1
	add %l2,1,%l2
	b L222
	nop
L223:
	cmp %l0,0
	bne L229
	nop
	cmp %l1,0
	bne L229
	nop
	b L228
	nop
L229:
	mov 0,%i0
	b L221
	nop
L228:
	mov %l2,%o0
	sll %o0,2,%o1
	add %o1,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%l3
	mov %l3,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	mov %o1,%l3
	ld [%fp+68],%l0
	mov 0,%l2
L230:
	cmp %l0,0
	be L231
	nop
	ld [%l0+20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %l2,%o1
	sll %o1,2,%o2
	st %o0,[%l3+%o2]
L232:
	ld [%l0+4],%l0
	add %l2,1,%l2
	b L230
	nop
L231:
	nop
	ld [%fp+68],%l0
	mov 0,%l2
L233:
	cmp %l0,0
	be L234
	nop
	st %g0,[%fp-20]
	ld [%fp+72],%l1
	mov 0,%l4
L236:
	cmp %l4,%l2
	bge L237
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	ld [%l1+64],%o0
	ld [%l3+%o1],%o1
	call _reg_mentioned_p,0
	nop
	cmp %o0,0
	be L239
	nop
	mov 1,%o0
	st %o0,[%fp-20]
	b L237
	nop
L239:
L238:
	ld [%l1+4],%l1
	add %l4,1,%l4
	b L236
	nop
L237:
	ld [%fp-20],%o0
	cmp %o0,0
	be L240
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	ld [%l3+%o1],%o0
	call _copy_to_reg,0
	nop
	mov %l2,%o1
	sll %o1,2,%o2
	st %o0,[%l3+%o2]
L240:
L235:
	ld [%l0+4],%l0
	add %l2,1,%l2
	b L233
	nop
L234:
	nop
	ld [%fp+72],%l1
	ld [%fp+68],%l0
	mov 0,%l2
L241:
	cmp %l1,0
	be L242
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	ld [%l3+%o1],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	ld [%l1+28],%o0
	cmp %o0,%o1
	bne L244
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	ld [%l1+64],%o0
	ld [%l3+%o1],%o1
	call _emit_move_insn,0
	nop
	b L245
	nop
L244:
	mov %l2,%o0
	sll %o0,2,%o1
	ld [%l0+20],%o0
	ld [%o0+8],%o2
	ld [%o2+12],%o3
	srl %o3,11,%o0
	and %o0,1,%o2
	ld [%l1+64],%o0
	ld [%l3+%o1],%o1
	call _convert_move,0
	nop
L245:
L243:
	ld [%l1+4],%l1
	ld [%l0+4],%l0
	add %l2,1,%l2
	b L241
	nop
L242:
	mov 1,%i0
	b L221
	nop
L221:
	ret
	restore
	.align 4
	.global _expand_start_bindings
	.proc	020
_expand_start_bindings:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov 40,%o0
	call _xmalloc,0
	nop
	st %o0,[%fp-20]
	mov 0,%o0
	mov -2,%o1
	call _emit_note,0
	nop
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	sethi %hi(_block_stack),%o1
	ld [%o1+%lo(_block_stack)],%o2
	st %o2,[%o0+4]
	ld [%fp-20],%o0
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o2
	st %o2,[%o0]
	ld [%fp-20],%o0
	sethi %hi(_nesting_depth),%o2
	sethi %hi(_nesting_depth),%o1
	sethi %hi(_nesting_depth),%o2
	ld [%o2+%lo(_nesting_depth)],%o3
	add %o3,1,%o2
	mov %o2,%o3
	st %o3,[%o1+%lo(_nesting_depth)]
	st %o3,[%o0+8]
	ld [%fp-20],%o0
	st %g0,[%o0+16]
	ld [%fp-20],%o0
	st %g0,[%o0+28]
	ld [%fp-20],%l0
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	cmp %o1,0
	be L247
	nop
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o2
	mov 0,%o0
	ld [%o1+28],%o1
	ld [%o2+32],%o2
	call _tree_cons,0
	nop
	b L248
	nop
L247:
	mov 0,%o0
L248:
	st %o0,[%l0+32]
	ld [%fp-20],%o0
	st %g0,[%o0+36]
	ld [%fp-20],%o0
	sethi %hi(_stack_block_stack),%o1
	ld [%o1+%lo(_stack_block_stack)],%o2
	st %o2,[%o0+24]
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	st %o1,[%o0+20]
	ld [%fp-20],%l0
	ld [%fp+68],%o0
	cmp %o0,0
	be L249
	nop
	call _gen_label_rtx,0
	nop
	b L250
	nop
L249:
	mov 0,%o0
L250:
	st %o0,[%l0+12]
	sethi %hi(_block_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_block_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_nesting_stack)]
L246:
	ret
	restore
	.align 4
	.global _use_variable
	.proc	020
_use_variable:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L252
	nop
	mov 26,%o0
	mov 0,%o1
	ld [%fp+68],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
	b L253
	nop
L252:
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L254
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L254
	nop
	ld [%fp+68],%o0
	sethi %hi(_frame_pointer_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_frame_pointer_rtx)],%o1
	cmp %o0,%o1
	be L254
	nop
	ld [%fp+68],%o0
	sethi %hi(_arg_pointer_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_arg_pointer_rtx)],%o1
	cmp %o0,%o1
	be L254
	nop
	ld [%fp+68],%o2
	mov 26,%o0
	mov 0,%o1
	ld [%o2+4],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_insn,0
	nop
L254:
L253:
L251:
	ret
	restore
	.align 4
	.proc	020
_use_variable_after:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L256
	nop
	mov 26,%o0
	mov 0,%o1
	ld [%fp+68],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+72],%o1
	call _emit_insn_after,0
	nop
	b L257
	nop
L256:
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L258
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L258
	nop
	ld [%fp+68],%o0
	sethi %hi(_frame_pointer_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_frame_pointer_rtx)],%o1
	cmp %o0,%o1
	be L258
	nop
	ld [%fp+68],%o0
	sethi %hi(_arg_pointer_rtx),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_arg_pointer_rtx)],%o1
	cmp %o0,%o1
	be L258
	nop
	ld [%fp+68],%o2
	mov 26,%o0
	mov 0,%o1
	ld [%o2+4],%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+72],%o1
	call _emit_insn_after,0
	nop
L258:
L257:
L255:
	ret
	restore
	.align 8
LC10:
	.ascii "unused variable `%s'\0"
	.align 4
	.global _expand_end_bindings
	.proc	020
_expand_end_bindings:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%l0
	sethi %hi(_warn_unused),%o0
	ld [%o0+%lo(_warn_unused)],%o1
	cmp %o1,0
	be L260
	nop
	ld [%fp+68],%l1
L261:
	cmp %l1,0
	be L262
	nop
	ld [%l1+12],%o1
	and %o1,256,%o0
	cmp %o0,0
	bne L264
	nop
	ldub [%l1+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	bne L264
	nop
	mov %l1,%o0
	sethi %hi(LC10),%o2
	or %o2,%lo(LC10),%o1
	call _warning_with_decl,0
	nop
L264:
L263:
	ld [%l1+4],%l1
	b L261
	nop
L262:
L260:
	ld [%fp+72],%o0
	cmp %o0,0
	be L265
	nop
	mov 0,%o0
	mov -3,%o1
	call _emit_note,0
	nop
	b L266
	nop
L265:
	ld [%l0+20],%o0
	mov -1,%o1
	st %o1,[%o0+20]
L266:
	ld [%l0+12],%o0
	cmp %o0,0
	be L267
	nop
	call _do_pending_stack_adjust,0
	nop
	ld [%l0+12],%o0
	call _emit_label,0
	nop
L267:
	ld [%fp+76],%o0
	cmp %o0,0
	bne L269
	nop
	ld [%l0+16],%o0
	cmp %o0,0
	bne L269
	nop
	ld [%l0+28],%o0
	cmp %o0,0
	bne L269
	nop
	b L268
	nop
L269:
	ld [%l0+36],%o0
	st %o0,[%fp-20]
L270:
	ld [%fp-20],%o0
	cmp %o0,0
	be L271
	nop
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	ld [%o0+12],%o1
	sethi %hi(524288),%o2
	or %o1,%o2,%o1
	st %o1,[%o0+12]
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	ld [%o1+12],%o0
	sethi %hi(16384),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	be L273
	nop
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	sethi %hi(LC1),%o2
	or %o2,%lo(LC1),%o1
	call _error_with_decl,0
	nop
L273:
L272:
	ld [%fp-20],%o0
	ld [%o0],%o1
	st %o1,[%fp-20]
	b L270
	nop
L271:
L268:
	ld [%l0+16],%o0
	cmp %o0,0
	bne L275
	nop
	ld [%l0+28],%o0
	cmp %o0,0
	bne L275
	nop
	b L274
	nop
L275:
	ld [%l0+28],%o0
	mov 0,%o1
	call _expand_cleanups,0
	nop
	ld [%l0+16],%o0
	cmp %o0,0
	be L276
	nop
	call _do_pending_stack_adjust,0
	nop
	sethi %hi(_stack_pointer_rtx),%o1
	ld [%o1+%lo(_stack_pointer_rtx)],%o0
	ld [%l0+16],%o1
	call _emit_move_insn,0
	nop
L276:
	mov %l0,%o0
	ld [%l0+16],%o1
	ld [%l0+28],%o2
	ld [%l0+20],%o3
	ld [%fp+76],%o4
	call _fixup_gotos,0
	nop
L274:
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	be L277
	nop
	ld [%fp+68],%l1
L278:
	cmp %l1,0
	be L279
	nop
	ld [%l1+64],%o0
	st %o0,[%fp-20]
	ldub [%l1+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	bne L281
	nop
	ld [%fp-20],%o0
	cmp %o0,0
	be L281
	nop
	ld [%fp-20],%o0
	call _use_variable,0
	nop
L281:
L280:
	ld [%l1+4],%l1
	b L278
	nop
L279:
L277:
	sethi %hi(_stack_block_stack),%o0
	ld [%l0+24],%o1
	st %o1,[%o0+%lo(_stack_block_stack)]
L282:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L285:
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-28]
	sethi %hi(_block_stack),%o0
	ld [%fp-28],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_block_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-28],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-28],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-28],%o0
	call _free,0
	nop
L287:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-24],%o0
	cmp %o1,%o0
	ble L286
	nop
	b L285
	nop
L286:
L284:
	b L283
	nop
	b L282
	nop
L283:
L259:
	ret
	restore
	.align 4
	.global _expand_decl
	.proc	020
_expand_decl:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
	ld [%fp+72],%o0
	cmp %o0,0
	be L289
	nop
	ld [%fp-20],%o2
	mov %l0,%o0
	ld [%fp+72],%o1
	ld [%o2+28],%o2
	call _temp_tree_cons,0
	nop
	ld [%fp-20],%o1
	st %o0,[%o1+28]
	sethi %hi(_stack_block_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_stack_block_stack)]
L289:
	cmp %l0,0
	bne L290
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	bne L291
	nop
	call _abort,0
	nop
L291:
	b L288
	nop
L290:
	ld [%l0+8],%o0
	st %o0,[%fp-24]
	ldub [%l0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	be L292
	nop
	b L288
	nop
L292:
	ld [%l0+12],%o0
	sethi %hi(2097152),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L294
	nop
	ld [%l0+12],%o0
	sethi %hi(8388608),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L294
	nop
	b L293
	nop
L294:
	b L288
	nop
L293:
	sethi %hi(_error_mark_node),%o0
	ld [%fp-24],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	bne L295
	nop
	sethi %hi(_const0_rtx),%o2
	mov 37,%o0
	mov 26,%o1
	ld [%o2+%lo(_const0_rtx)],%o2
	call _gen_rtx,0
	nop
	st %o0,[%l0+64]
	b L296
	nop
L295:
	ld [%l0+24],%o0
	cmp %o0,0
	bne L297
	nop
	ld [%l0+60],%o0
	cmp %o0,0
	bne L298
	nop
	ld [%l0+28],%o0
	mov 0,%o1
	call _assign_stack_local,0
	nop
	st %o0,[%l0+64]
	b L299
	nop
L298:
	mov 4,%o0
	call _gen_reg_rtx,0
	nop
	mov %o0,%o2
	mov 37,%o0
	mov 26,%o1
	call _gen_rtx,0
	nop
	st %o0,[%l0+64]
L299:
	b L300
	nop
L297:
	ld [%l0+28],%o0
	cmp %o0,26
	be L301
	nop
	sethi %hi(_flag_float_store),%o0
	ld [%o0+%lo(_flag_float_store)],%o1
	cmp %o1,0
	be L302
	nop
	ld [%fp-24],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,6
	bne L302
	nop
	b L301
	nop
L302:
	ld [%l0+12],%o0
	sethi %hi(1048576),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L301
	nop
	ld [%l0+12],%o0
	sethi %hi(16384),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L301
	nop
	ld [%l0+12],%o0
	sethi %hi(8192),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L303
	nop
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	bne L301
	nop
	b L303
	nop
L303:
	ld [%l0+28],%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%l0+64]
	ld [%fp-24],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,11
	bne L304
	nop
	ld [%l0+64],%o0
	call _mark_reg_pointer,0
	nop
L304:
	ld [%l0+64],%o0
	ld [%o0],%o1
	or %o1,16,%o2
	st %o2,[%o0]
	b L305
	nop
L301:
	ld [%l0+24],%o0
	ld [%o0+12],%o1
	sethi %hi(131072),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	be L306
	nop
	st %g0,[%fp-28]
	ld [%l0+64],%o0
	cmp %o0,0
	be L307
	nop
	ld [%l0+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L309
	nop
	ld [%l0+64],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L309
	nop
	b L308
	nop
L309:
	call _abort,0
	nop
L308:
	ld [%l0+64],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
L307:
	ld [%l0+24],%o1
	ldub [%l0+32],%o0
	and %o0,0xff,%o2
	mov %o2,%o0
	ld [%o1+16],%o1
	call .umul,0
	nop
	add %o0,7,%o1
	mov %o1,%o0
	cmp %o0,0
	bge L310
	nop
	add %o0,7,%o0
L310:
	sra %o0,3,%o1
	ld [%l0+28],%o0
	call _assign_stack_local,0
	nop
	st %o0,[%l0+64]
	ld [%fp-28],%o0
	cmp %o0,0
	be L311
	nop
	ld [%l0+64],%o1
	ld [%o1+4],%o0
	ld [%fp-28],%o1
	call _force_operand,0
	nop
	st %o0,[%fp-32]
	ld [%fp-28],%o0
	ld [%fp-32],%o1
	call _emit_move_insn,0
	nop
L311:
	ld [%l0+64],%o0
	mov 0,%o1
	ld [%l0+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o2
	cmp %o2,16
	be L313
	nop
	ld [%l0+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o2
	cmp %o2,19
	be L313
	nop
	ld [%l0+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o2
	cmp %o2,20
	be L313
	nop
	b L312
	nop
L313:
	mov 1,%o1
L312:
	and %o1,1,%o2
	sll %o2,3,%o1
	ld [%o0],%o3
	and %o3,-9,%o2
	or %o2,%o1,%o2
	st %o2,[%o0]
	b L314
	nop
L306:
	sethi %hi(_frame_pointer_needed),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_frame_pointer_needed)]
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	bne L315
	nop
	call _do_pending_stack_adjust,0
	nop
	sethi %hi(_stack_pointer_rtx),%o1
	ld [%o1+%lo(_stack_pointer_rtx)],%o0
	call _copy_to_reg,0
	nop
	ld [%fp-20],%o1
	st %o0,[%o1+16]
	sethi %hi(_stack_block_stack),%o0
	ld [%fp-20],%o1
	st %o1,[%o0+%lo(_stack_block_stack)]
L315:
	ldub [%l0+32],%o0
	and %o0,0xff,%o1
	ld [%l0+24],%o0
	mov 8,%o2
	call _convert_units,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	st %o0,[%fp-40]
	ldub [%l0+32],%o1
	and %o1,0xff,%o0
	and %o0,15,%o1
	and %o1,0xff,%o0
	cmp %o0,0
	be L316
	nop
	ld [%fp-40],%o0
	call _round_push,0
	nop
	st %o0,[%fp-40]
L316:
	ld [%fp-40],%o0
	call _anti_adjust_stack,0
	nop
	sethi %hi(_stack_pointer_rtx),%o1
	ld [%o1+%lo(_stack_pointer_rtx)],%o0
	call _copy_to_reg,0
	nop
	st %o0,[%fp-36]
	mov 37,%o0
	ld [%l0+28],%o1
	ld [%fp-36],%o2
	call _gen_rtx,0
	nop
	st %o0,[%l0+64]
L314:
L305:
L300:
L296:
	ld [%l0+12],%o0
	sethi %hi(1048576),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	be L317
	nop
	ld [%l0+64],%o0
	ld [%o0],%o1
	or %o1,16,%o2
	st %o2,[%o0]
L317:
	ld [%l0+12],%o0
	sethi %hi(262144),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	be L318
	nop
	ld [%l0+64],%o0
	ld [%o0],%o1
	or %o1,32,%o2
	st %o2,[%o0]
L318:
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	be L319
	nop
	ld [%l0+64],%o0
	call _use_variable,0
	nop
L319:
L288:
	ret
	restore
	.align 4
	.global _expand_decl_init
	.proc	020
_expand_decl_init:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	sethi %hi(2097152),%o2
	and %o1,%o2,%o0
	cmp %o0,0
	be L321
	nop
	b L320
	nop
L321:
	ld [%fp+68],%o0
	sethi %hi(_error_mark_node),%o1
	ld [%o0+60],%o0
	ld [%o1+%lo(_error_mark_node)],%o1
	cmp %o0,%o1
	bne L322
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ldub [%o1+12],%o0
	and %o0,0xff,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,5
	be L324
	nop
	ld [%fp-20],%o0
	cmp %o0,6
	be L324
	nop
	ld [%fp-20],%o0
	cmp %o0,8
	be L324
	nop
	ld [%fp-20],%o0
	cmp %o0,11
	be L324
	nop
	b L323
	nop
L324:
	ld [%fp+68],%o0
	sethi %hi(_integer_zero_node),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_integer_zero_node)],%o1
	call _convert,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	call _expand_assignment,0
	nop
L323:
	call _emit_queue,0
	nop
	b L325
	nop
L322:
	ld [%fp+68],%o0
	ld [%o0+60],%o1
	cmp %o1,0
	be L326
	nop
	ld [%fp+68],%o0
	ld [%o0+60],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,3
	be L326
	nop
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+16],%o0
	ld [%o1+20],%o1
	call _emit_line_note,0
	nop
	ld [%fp+68],%o1
	ld [%fp+68],%o0
	ld [%o1+60],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_assignment,0
	nop
	call _emit_queue,0
	nop
L326:
L325:
L320:
	ret
	restore
	.align 4
	.global _expand_anon_union_decl
	.proc	020
_expand_anon_union_decl:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call _expand_decl,0
	nop
	ld [%fp+68],%o0
	ld [%o0+64],%o1
	st %o1,[%fp-24]
L328:
	ld [%fp+76],%o0
	cmp %o0,0
	be L329
	nop
	ld [%fp+76],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-28]
	ld [%fp+76],%o0
	ld [%o0+16],%o1
	st %o1,[%fp-32]
	ld [%fp-28],%l0
	ld [%fp-24],%o0
	ldub [%o0+2],%o1
	and %o1,0xff,%o0
	cmp %o0,26
	be L330
	nop
	ld [%fp-28],%o0
	ld [%o0+8],%o1
	ldub [%o1+28],%o0
	and %o0,0xff,%o1
	mov 35,%o0
	ld [%fp-24],%o2
	mov 0,%o3
	call _gen_rtx,0
	nop
	b L331
	nop
L330:
	ld [%fp-24],%o0
L331:
	st %o0,[%l0+64]
	ld [%fp+72],%o0
	cmp %o0,0
	be L332
	nop
	ld [%fp-20],%o2
	ld [%fp-28],%o0
	ld [%fp-32],%o1
	ld [%o2+28],%o2
	call _temp_tree_cons,0
	nop
	ld [%fp-20],%o1
	st %o0,[%o1+28]
L332:
	ld [%fp+76],%o0
	ld [%o0+4],%o1
	st %o1,[%fp+76]
	b L328
	nop
L329:
L327:
	ret
	restore
	.align 4
	.proc	020
_expand_cleanups:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L334:
	ld [%fp-20],%o0
	cmp %o0,0
	be L335
	nop
	ld [%fp+72],%o0
	cmp %o0,0
	be L338
	nop
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	ld [%fp+72],%o0
	cmp %o1,%o0
	bne L338
	nop
	b L337
	nop
L338:
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,3
	bne L339
	nop
	ld [%fp-20],%o1
	ld [%o1+20],%o0
	ld [%fp+72],%o1
	call _expand_cleanups,0
	nop
	b L340
	nop
L339:
	ld [%fp-20],%o0
	sethi %hi(_const0_rtx),%o1
	ld [%o0+20],%o0
	ld [%o1+%lo(_const0_rtx)],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
L340:
L337:
L336:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L334
	nop
L335:
L333:
	ret
	restore
	.align 4
	.proc	020
_fixup_cleanups:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	call _get_last_insn,0
	nop
	st %o0,[%fp-20]
	ld [%fp+68],%o0
	mov 0,%o1
	call _expand_cleanups,0
	nop
	call _get_last_insn,0
	nop
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	ld [%fp+72],%o2
	ld [%o0+12],%o0
	ld [%fp-24],%o1
	ld [%o2],%o2
	call _reorder_insns,0
	nop
	ld [%fp+72],%o0
	ld [%fp-24],%o1
	st %o1,[%o0]
L341:
	ret
	restore
	.align 4
	.global _move_cleanups_up
	.proc	020
_move_cleanups_up:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	sethi %hi(_block_stack),%o0
	ld [%o0+%lo(_block_stack)],%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	ld [%o0+28],%o0
	ld [%o1+28],%o1
	call _chainon,0
	nop
	ld [%fp-24],%o1
	st %o0,[%o1+28]
	ld [%fp-20],%o0
	st %g0,[%o0+28]
L342:
	ret
	restore
	.align 4
	.global _expand_start_case
	.proc	020
_expand_start_case:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	mov 40,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	st %o1,[%l0+4]
	sethi %hi(_nesting_stack),%o0
	ld [%o0+%lo(_nesting_stack)],%o1
	st %o1,[%l0]
	sethi %hi(_nesting_depth),%o1
	sethi %hi(_nesting_depth),%o0
	sethi %hi(_nesting_depth),%o1
	ld [%o1+%lo(_nesting_depth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	st %o2,[%l0+8]
	ld [%fp+68],%o0
	cmp %o0,0
	be L344
	nop
	call _gen_label_rtx,0
	nop
	b L345
	nop
L344:
	mov 0,%o0
L345:
	st %o0,[%l0+12]
	st %g0,[%l0+20]
	ld [%fp+72],%o0
	st %o0,[%l0+28]
	ld [%fp+76],%o0
	st %o0,[%l0+32]
	st %g0,[%l0+24]
	sth %g0,[%l0+36]
	sethi %hi(_case_stack),%o0
	st %l0,[%o0+%lo(_case_stack)]
	sethi %hi(_nesting_stack),%o0
	st %l0,[%o0+%lo(_nesting_stack)]
	call _do_pending_stack_adjust,0
	nop
	call _get_last_insn,0
	nop
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,18
	be L346
	nop
	mov 0,%o0
	mov -1,%o1
	call _emit_note,0
	nop
L346:
	call _get_last_insn,0
	nop
	st %o0,[%l0+16]
L343:
	ret
	restore
	.align 4
	.global _expand_start_case_dummy
	.proc	020
_expand_start_case_dummy:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov 40,%o0
	call _xmalloc,0
	nop
	mov %o0,%l0
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	st %o1,[%l0+4]
	sethi %hi(_nesting_stack),%o0
	ld [%o0+%lo(_nesting_stack)],%o1
	st %o1,[%l0]
	sethi %hi(_nesting_depth),%o1
	sethi %hi(_nesting_depth),%o0
	sethi %hi(_nesting_depth),%o1
	ld [%o1+%lo(_nesting_depth)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	st %o2,[%l0+8]
	st %g0,[%l0+12]
	st %g0,[%l0+20]
	st %g0,[%l0+16]
	st %g0,[%l0+32]
	st %g0,[%l0+24]
	sth %g0,[%l0+36]
	sethi %hi(_case_stack),%o0
	st %l0,[%o0+%lo(_case_stack)]
	sethi %hi(_nesting_stack),%o0
	st %l0,[%o0+%lo(_nesting_stack)]
L347:
	ret
	restore
	.align 4
	.global _expand_end_case_dummy
	.proc	020
_expand_end_case_dummy:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	nop
L349:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-20]
L352:
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	st %o1,[%fp-24]
	sethi %hi(_case_stack),%o0
	ld [%fp-24],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_case_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-24],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-24],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-24],%o0
	call _free,0
	nop
L354:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-20],%o0
	cmp %o1,%o0
	ble L353
	nop
	b L352
	nop
L353:
L351:
	b L350
	nop
	b L349
	nop
L350:
L348:
	ret
	restore
	.align 4
	.global _pushcase
	.proc	04
_pushcase:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l1
	mov %i1,%l0
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	cmp %o1,0
	be L357
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	bne L356
	nop
	b L357
	nop
L357:
	mov 1,%i0
	b L355
	nop
L356:
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+28],%o1
	ld [%o1+8],%o0
	st %o0,[%fp-20]
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+32],%o1
	st %o1,[%fp-24]
	sethi %hi(_error_mark_node),%o0
	ld [%fp-20],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	bne L358
	nop
	mov 0,%i0
	b L355
	nop
L358:
	cmp %l1,0
	be L359
	nop
	ld [%fp-24],%o0
	mov %l1,%o1
	call _convert,0
	nop
	mov %o0,%l1
L359:
	cmp %l1,0
	be L360
	nop
	mov %l1,%o0
	ld [%fp-20],%o1
	call _int_fits_type_p,0
	nop
	cmp %o0,0
	bne L360
	nop
	mov 3,%i0
	b L355
	nop
L360:
	cmp %l1,0
	bne L361
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+24],%o1
	cmp %o1,0
	be L362
	nop
	mov 2,%i0
	b L355
	nop
L362:
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	st %l0,[%o0+24]
	b L363
	nop
L361:
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	add %o1,20,%l2
L364:
	ld [%l2],%o0
	cmp %o0,0
	be L365
	nop
	ld [%l2],%o1
	ld [%o1+16],%o0
	mov %l1,%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L365
	nop
L366:
	ld [%l2],%o0
	add %o0,4,%l2
	b L364
	nop
L365:
	ld [%l2],%o0
	cmp %o0,0
	be L367
	nop
	ld [%l2],%o1
	mov %l1,%o0
	ld [%o1+12],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	bne L368
	nop
	mov 2,%i0
	b L355
	nop
L368:
L367:
	mov 28,%o0
	call _oballoc,0
	nop
	mov %o0,%l3
	st %g0,[%l3]
	ld [%l2],%o0
	st %o0,[%l3+4]
	mov %l1,%o0
	call _copy_node,0
	nop
	mov %o0,%o1
	st %o1,[%l3+12]
	st %o1,[%l3+16]
	st %l0,[%l3+24]
	st %g0,[%l3+20]
	st %l3,[%l2]
L363:
	mov %l0,%o0
	call _expand_label,0
	nop
	mov 0,%i0
	b L355
	nop
L355:
	ret
	restore
	.align 4
	.global _pushcase_range
	.proc	04
_pushcase_range:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	mov %i0,%l2
	mov %i1,%l1
	mov %i2,%l0
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	cmp %o1,0
	be L371
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	bne L370
	nop
	b L371
	nop
L371:
	mov 1,%i0
	b L369
	nop
L370:
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+28],%o1
	ld [%o1+8],%o0
	st %o0,[%fp-20]
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+32],%o1
	st %o1,[%fp-24]
	sethi %hi(_error_mark_node),%o0
	ld [%fp-20],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	bne L372
	nop
	mov 0,%i0
	b L369
	nop
L372:
	cmp %l2,0
	be L373
	nop
	ld [%fp-24],%o0
	mov %l2,%o1
	call _convert,0
	nop
	mov %o0,%l2
L373:
	cmp %l1,0
	be L374
	nop
	ld [%fp-24],%o0
	mov %l1,%o1
	call _convert,0
	nop
	mov %o0,%l1
L374:
	cmp %l2,0
	be L375
	nop
	mov %l2,%o0
	ld [%fp-20],%o1
	call _int_fits_type_p,0
	nop
	cmp %o0,0
	bne L375
	nop
	mov 3,%i0
	b L369
	nop
L375:
	cmp %l1,0
	be L376
	nop
	mov %l1,%o0
	ld [%fp-20],%o1
	call _int_fits_type_p,0
	nop
	cmp %o0,0
	bne L376
	nop
	mov 3,%i0
	b L369
	nop
L376:
	mov %l1,%o0
	mov %l2,%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L377
	nop
	mov 4,%i0
	b L369
	nop
L377:
	mov %l2,%o0
	mov %l1,%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	be L378
	nop
	mov %l2,%o0
	mov %l0,%o1
	call _pushcase,0
	nop
	mov %o0,%i0
	b L369
	nop
L378:
	nop
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	add %o1,20,%l3
L379:
	ld [%l3],%o0
	cmp %o0,0
	be L380
	nop
	ld [%l3],%o1
	ld [%o1+16],%o0
	mov %l2,%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L380
	nop
L381:
	ld [%l3],%o0
	add %o0,4,%l3
	b L379
	nop
L380:
	ld [%l3],%o0
	cmp %o0,0
	be L382
	nop
	ld [%l3],%o1
	mov %l1,%o0
	ld [%o1+12],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	bne L383
	nop
	mov 2,%i0
	b L369
	nop
L383:
L382:
	mov 28,%o0
	call _oballoc,0
	nop
	mov %o0,%l4
	st %g0,[%l4]
	ld [%l3],%o0
	st %o0,[%l4+4]
	mov %l2,%o0
	call _copy_node,0
	nop
	st %o0,[%l4+12]
	mov %l1,%o0
	call _copy_node,0
	nop
	st %o0,[%l4+16]
	st %l0,[%l4+24]
	st %g0,[%l4+20]
	st %l4,[%l3]
	mov %l0,%o0
	call _expand_label,0
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	sethi %hi(_case_stack),%o2
	ld [%o2+%lo(_case_stack)],%o1
	lduh [%o1+36],%o2
	add %o2,1,%o1
	mov %o1,%o2
	sth %o2,[%o0+36]
	mov 0,%i0
	b L369
	nop
L369:
	ret
	restore
	.align 8
LC11:
	.ascii "enumerated value `%s' not handled in switch\0"
	.align 8
LC12:
	.ascii "case value `%d' not in enumerated type `%s'\0"
	.align 4
	.global _check_for_full_enumeration_handling
	.proc	020
_check_for_full_enumeration_handling:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+28],%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L385
	nop
	b L384
	nop
	b L386
	nop
L385:
	ld [%fp-20],%o0
	ld [%o0+20],%o1
	st %o1,[%fp-24]
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	ld [%o1+16],%l1
L387:
	cmp %l1,0
	be L388
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+20],%l0
L390:
	cmp %l0,0
	be L391
	nop
	ld [%l0+16],%o0
	ld [%l1+20],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L391
	nop
L392:
	ld [%l0+4],%l0
	b L390
	nop
L391:
	cmp %l0,0
	be L394
	nop
	ld [%l0+12],%o0
	ld [%l1+20],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	bne L393
	nop
	b L394
	nop
L394:
	ld [%l1+16],%o1
	sethi %hi(LC11),%o2
	or %o2,%lo(LC11),%o0
	ld [%o1+20],%o1
	call _warning,0
	nop
L393:
L389:
	ld [%l1+4],%l1
	b L387
	nop
L388:
	nop
	sethi %hi(_case_stack),%o1
	ld [%o1+%lo(_case_stack)],%o0
	ld [%o0+20],%l0
L395:
	cmp %l0,0
	be L396
	nop
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	ld [%o1+16],%l1
L398:
	cmp %l1,0
	be L399
	nop
	ld [%l0+12],%o0
	ld [%l1+20],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	bne L399
	nop
L400:
	ld [%l1+4],%l1
	b L398
	nop
L399:
	cmp %l1,0
	bne L401
	nop
	ld [%l0+12],%o1
	ld [%fp-24],%o0
	ld [%o0+8],%o2
	ld [%o2+48],%o0
	ld [%o0+36],%o2
	sethi %hi(LC12),%o3
	or %o3,%lo(LC12),%o0
	ld [%o1+16],%o1
	ld [%o2+20],%o2
	call _warning,0
	nop
L401:
L397:
	ld [%l0+4],%l0
	b L395
	nop
L396:
L386:
L384:
	ret
	restore
	.align 4
	.global _expand_end_case
	.proc	020
_expand_end_case:
	!#PROLOGUE# 0
	save %sp,-168,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %g0,[%fp-32]
	call _gen_label_rtx,0
	nop
	st %o0,[%fp-44]
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%l2
	ld [%l2+28],%o0
	st %o0,[%fp-60]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-60],%o0
	sethi %hi(_error_mark_node),%o1
	ld [%o0+8],%o0
	ld [%o1+%lo(_error_mark_node)],%o1
	cmp %o0,%o1
	be L403
	nop
	ld [%l2+24],%o0
	cmp %o0,0
	bne L404
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,8
	bne L404
	nop
	sethi %hi(_warn_switch),%o0
	ld [%o0+%lo(_warn_switch)],%o1
	cmp %o1,0
	be L404
	nop
	call _check_for_full_enumeration_handling,0
	nop
L404:
	ld [%l2+24],%o0
	cmp %o0,0
	bne L405
	nop
	mov 40,%o0
	mov 0,%o1
	mov 0,%o2
	call _build_decl,0
	nop
	st %o0,[%l2+24]
	ld [%l2+24],%o0
	call _expand_label,0
	nop
L405:
	ld [%l2+24],%o0
	call _label_rtx,0
	nop
	st %o0,[%fp-32]
	call _get_last_insn,0
	nop
	st %o0,[%fp-56]
	ld [%l2+20],%o0
	call _group_case_nodes,0
	nop
	st %g0,[%fp-36]
	ld [%l2+20],%l0
L406:
	cmp %l0,0
	be L407
	nop
	ld [%l0+12],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L409
	nop
	call _abort,0
	nop
L409:
	ld [%l0+16],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L410
	nop
	call _abort,0
	nop
L410:
	ld [%fp-60],%o1
	ld [%o1+8],%o0
	ld [%l0+12],%o1
	call _convert,0
	nop
	st %o0,[%l0+12]
	ld [%fp-60],%o1
	ld [%o1+8],%o0
	ld [%l0+16],%o1
	call _convert,0
	nop
	st %o0,[%l0+16]
	ld [%fp-36],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-36]
	cmp %o1,1
	bne L411
	nop
	ld [%l0+12],%o0
	st %o0,[%fp-20]
	ld [%l0+16],%o0
	st %o0,[%fp-24]
	b L412
	nop
L411:
	ld [%l0+12],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bl L414
	nop
	ld [%l0+12],%o0
	ld [%fp-20],%o1
	ld [%o0+20],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bne L413
	nop
	ld [%l0+12],%o0
	ld [%fp-20],%o1
	ld [%o0+16],%o0
	ld [%o1+16],%o1
	cmp %o0,%o1
	blu L414
	nop
	b L413
	nop
L414:
	ld [%l0+12],%o0
	st %o0,[%fp-20]
L413:
	ld [%fp-24],%o0
	ld [%l0+16],%o1
	ld [%o0+20],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bl L416
	nop
	ld [%fp-24],%o0
	ld [%l0+16],%o1
	ld [%o0+20],%o0
	ld [%o1+20],%o1
	cmp %o0,%o1
	bne L415
	nop
	ld [%fp-24],%o0
	ld [%l0+16],%o1
	ld [%o0+16],%o0
	ld [%o1+16],%o1
	cmp %o0,%o1
	blu L416
	nop
	b L415
	nop
L416:
	ld [%l0+16],%o0
	st %o0,[%fp-24]
L415:
L412:
	ld [%l0+12],%o0
	ld [%l0+16],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	bne L417
	nop
	ld [%fp-36],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-36]
L417:
L408:
	ld [%l0+4],%l0
	b L406
	nop
L407:
	ld [%fp-36],%o0
	cmp %o0,0
	be L418
	nop
	mov 64,%o0
	ld [%fp-24],%o1
	ld [%fp-20],%o2
	call _combine,0
	nop
	st %o0,[%fp-28]
L418:
	ld [%fp-36],%o0
	cmp %o0,0
	be L420
	nop
	ld [%fp-60],%o0
	ld [%o0+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	bne L419
	nop
	b L420
	nop
L420:
	sethi %hi(_const0_rtx),%o1
	ld [%fp-60],%o0
	ld [%o1+%lo(_const0_rtx)],%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	call _emit_queue,0
	nop
	ld [%fp-32],%o0
	call _emit_jump,0
	nop
	b L421
	nop
L419:
	ld [%fp-28],%o0
	ld [%o0+20],%o1
	cmp %o1,0
	bne L423
	nop
	ld [%fp-36],%o0
	cmp %o0,3
	ble L423
	nop
	ld [%fp-28],%o0
	ld [%fp-36],%o1
	mov %o1,%o3
	sll %o3,2,%o2
	add %o2,%o1,%o2
	sll %o2,1,%o1
	ld [%o0+16],%o0
	cmp %o0,%o1
	bgu L423
	nop
	ld [%fp-60],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L423
	nop
	b L422
	nop
L423:
	ld [%fp-60],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	st %o0,[%fp-40]
	call _emit_queue,0
	nop
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-40],%o0
	mov 0,%o1
	call _protect_from_queue,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L424
	nop
	ld [%fp-40],%o0
	call _copy_to_reg,0
	nop
	st %o0,[%fp-40]
L424:
	ld [%fp-40],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	be L426
	nop
	ld [%fp-60],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L426
	nop
	b L425
	nop
L426:
	ld [%fp-60],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	be L427
	nop
	ld [%fp-40],%o1
	ld [%o1+4],%o0
	mov 0,%o1
	call _build_int_2,0
	nop
	st %o0,[%fp-60]
	ld [%fp-60],%o1
	ld [%o1+8],%o0
	ld [%fp-60],%o1
	call _convert,0
	nop
	st %o0,[%fp-60]
L427:
	nop
	ld [%l2+20],%l0
L428:
	cmp %l0,0
	be L429
	nop
	ld [%fp-60],%o0
	ld [%l0+12],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	bne L431
	nop
	ld [%l0+16],%o0
	ld [%fp-60],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	bne L431
	nop
	b L429
	nop
L431:
L430:
	ld [%l0+4],%l0
	b L428
	nop
L429:
	cmp %l0,0
	be L432
	nop
	ld [%l0+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L433
	nop
L432:
	ld [%fp-32],%o0
	call _emit_jump,0
	nop
L433:
	b L434
	nop
L425:
	add %l2,20,%o1
	mov %o1,%o0
	mov 0,%o1
	call _balance_case_nodes,0
	nop
	ld [%fp-60],%o0
	ld [%o0+8],%o1
	ld [%o1+12],%o2
	srl %o2,11,%o0
	and %o0,1,%o3
	ld [%fp-40],%o0
	ld [%l2+20],%o1
	ld [%fp-32],%o2
	call _emit_case_nodes,0
	nop
	ld [%fp-32],%o0
	call _emit_jump_if_reachable,0
	nop
L434:
	b L435
	nop
L422:
	ld [%fp-60],%o0
	ld [%o0+8],%o1
	ldub [%o1+28],%o2
	and %o2,0xff,%o0
	cmp %o0,6
	bne L436
	nop
	ld [%fp-60],%o1
	mov 64,%o0
	ld [%o1+8],%o1
	ld [%fp-60],%o2
	ld [%fp-20],%o3
	call _build,0
	nop
	st %o0,[%fp-60]
	sethi %hi(_integer_zero_node),%o0
	ld [%o0+%lo(_integer_zero_node)],%o1
	st %o1,[%fp-20]
L436:
	ld [%fp-60],%o0
	ld [%o0+8],%o1
	ldub [%o1+28],%o2
	and %o2,0xff,%o0
	cmp %o0,4
	be L437
	nop
	sethi %hi(_mode_size+16),%o0
	ld [%o0+%lo(_mode_size+16)],%o1
	mov %o1,%o0
	sll %o0,3,%o1
	mov %o1,%o0
	mov 0,%o1
	call _type_for_size,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-60],%o1
	call _convert,0
	nop
	st %o0,[%fp-60]
L437:
	ld [%fp-60],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	st %o0,[%fp-40]
	call _emit_queue,0
	nop
	ld [%fp-40],%o0
	mov 0,%o1
	call _protect_from_queue,0
	nop
	st %o0,[%fp-40]
	call _do_pending_stack_adjust,0
	nop
	ld [%fp-20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%l3
	ld [%fp-28],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o2
	ld [%fp-40],%o0
	mov %l3,%o1
	ld [%fp-44],%o3
	ld [%fp-32],%o4
	call _gen_casesi,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp-28],%o0
	ld [%o0+16],%o1
	add %o1,1,%o0
	st %o0,[%fp-48]
	ld [%fp-48],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	add %o2,7,%o0
	add %sp,92,%o2
	sub %o2,%sp,%o1
	add %o0,%o1,%o2
	mov %o2,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	sub %sp,%o1,%sp
	add %sp,92,%o1
	mov %o1,%o0
	add %o0,7,%o0
	srl %o0,3,%o1
	mov %o1,%o0
	sll %o0,3,%o1
	st %o1,[%fp-52]
	ld [%fp-48],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	ld [%fp-52],%o0
	mov 0,%o1
	call _memset,0
	nop
	ld [%l2+20],%l0
L438:
	cmp %l0,0
	be L439
	nop
	ld [%l0+12],%o0
	ld [%fp-20],%o1
	ld [%o0+16],%o0
	ld [%o1+16],%o1
	sub %o0,%o1,%l3
L441:
	ld [%fp-20],%o0
	ld [%o0+16],%o1
	add %l3,%o1,%o0
	ld [%l0+16],%o1
	ld [%o1+16],%o2
	cmp %o0,%o2
	bg L442
	nop
	ld [%l0+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o2
	mov 38,%o0
	mov 4,%o1
	call _gen_rtx,0
	nop
	mov %l3,%o1
	mov %o1,%o2
	sll %o2,2,%o1
	ld [%fp-52],%o2
	add %o1,%o2,%o1
	st %o0,[%o1]
	add %l3,1,%l3
	b L441
	nop
L442:
L440:
	ld [%l0+4],%l0
	b L438
	nop
L439:
	nop
	mov 0,%l1
L443:
	ld [%fp-48],%o0
	cmp %l1,%o0
	bge L444
	nop
	mov %l1,%o0
	sll %o0,2,%o1
	ld [%fp-52],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	cmp %o1,0
	bne L446
	nop
	mov 38,%o0
	mov 4,%o1
	ld [%fp-32],%o2
	call _gen_rtx,0
	nop
	mov %l1,%o1
	sll %o1,2,%o2
	ld [%fp-52],%o3
	add %o2,%o3,%o1
	st %o0,[%o1]
L446:
L445:
	add %l1,1,%l1
	b L443
	nop
L444:
	ld [%fp-44],%o0
	call _emit_label,0
	nop
	mov 38,%o0
	mov 4,%o1
	ld [%fp-44],%o2
	call _gen_rtx,0
	nop
	mov %o0,%l3
	ld [%fp-48],%o0
	ld [%fp-52],%o1
	call _gen_rtvec_v,0
	nop
	mov %o0,%o3
	mov 24,%o0
	mov 2,%o1
	mov %l3,%o2
	call _gen_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	call _emit_barrier,0
	nop
L435:
L421:
	ld [%fp-56],%l3
	call _get_last_insn,0
	nop
	mov %o0,%o1
	ld [%l3+12],%o0
	ld [%l2+16],%o2
	call _reorder_insns,0
	nop
L403:
	ld [%l2+12],%o0
	cmp %o0,0
	be L447
	nop
	ld [%l2+12],%o0
	call _emit_label,0
	nop
L447:
	nop
L448:
	sethi %hi(_nesting_stack),%o1
	ld [%o1+%lo(_nesting_stack)],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-64]
L451:
	sethi %hi(_case_stack),%o0
	ld [%o0+%lo(_case_stack)],%o1
	st %o1,[%fp-68]
	sethi %hi(_case_stack),%o0
	ld [%fp-68],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_case_stack)]
	sethi %hi(_nesting_stack),%o0
	ld [%fp-68],%o1
	ld [%o1],%o2
	st %o2,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	ld [%fp-68],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_nesting_depth)]
	ld [%fp-68],%o0
	call _free,0
	nop
L453:
	sethi %hi(_nesting_depth),%o0
	ld [%o0+%lo(_nesting_depth)],%o1
	ld [%fp-64],%o0
	cmp %o1,%o0
	ble L452
	nop
	b L451
	nop
L452:
L450:
	b L449
	nop
	b L448
	nop
L449:
L402:
	ret
	restore
	.align 4
	.proc	020
_do_jump_if_equal:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L455
	nop
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L455
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	cmp %o0,%o1
	bne L456
	nop
	ld [%fp+76],%o0
	call _emit_jump,0
	nop
L456:
	b L457
	nop
L455:
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	mov 0,%o2
	ld [%fp+80],%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+76],%o0
	call _gen_beq,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L457:
L454:
	ret
	restore
	.align 4
	.proc	020
_group_case_nodes:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	st %o0,[%fp-20]
L459:
	ld [%fp-20],%o0
	cmp %o0,0
	be L460
	nop
	ld [%fp-20],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _next_real_insn,0
	nop
	st %o0,[%fp-24]
	ld [%fp-20],%o0
	st %o0,[%fp-28]
L461:
	ld [%fp-28],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	cmp %o1,0
	be L462
	nop
	ld [%fp-28],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _next_real_insn,0
	nop
	ld [%fp-24],%o1
	cmp %o0,%o1
	bne L462
	nop
	ld [%fp-28],%l0
	ld [%fp-20],%l1
	mov 1,%o0
	mov 0,%o1
	call _build_int_2,0
	nop
	mov %o0,%o2
	mov 63,%o0
	ld [%l1+16],%o1
	call _combine,0
	nop
	mov %o0,%o1
	ld [%l0+12],%o0
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	be L462
	nop
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	ld [%o1+16],%o2
	st %o2,[%o0+16]
	b L461
	nop
L462:
	ld [%fp-20],%o0
	ld [%fp-28],%o1
	st %o1,[%o0+4]
	ld [%fp-28],%o0
	st %o0,[%fp-20]
	b L459
	nop
L460:
L458:
	ret
	restore
	.align 4
	.proc	020
_balance_case_nodes:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0],%l0
	cmp %l0,0
	be L464
	nop
	st %g0,[%fp-20]
	st %g0,[%fp-24]
L465:
	cmp %l0,0
	be L466
	nop
	ld [%l0+12],%o0
	ld [%l0+16],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	bne L467
	nop
	ld [%fp-24],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-24]
L467:
	ld [%fp-20],%o1
	add %o1,1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	ld [%l0+4],%l0
	b L465
	nop
L466:
	ld [%fp-20],%o0
	cmp %o0,2
	ble L468
	nop
	ld [%fp+68],%l1
	ld [%l1],%o0
	st %o0,[%fp-28]
	ld [%fp-20],%o0
	cmp %o0,3
	bne L469
	nop
	ld [%l1],%o0
	add %o0,4,%l1
	b L470
	nop
L469:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	add %o0,%o1,%o0
	add %o0,1,%o1
	mov %o1,%o0
	srl %o0,31,%o1
	add %o0,%o1,%o0
	sra %o0,1,%o1
	st %o1,[%fp-20]
L471:
	ld [%l1],%o0
	ld [%l1],%o1
	ld [%o0+12],%o0
	ld [%o1+16],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	bne L473
	nop
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
L473:
	ld [%fp-20],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-20]
	ld [%fp-20],%o0
	cmp %o0,0
	bg L474
	nop
	b L472
	nop
L474:
	ld [%l1],%o0
	add %o0,4,%l1
	b L471
	nop
L472:
L470:
	ld [%fp+68],%o0
	ld [%l1],%l0
	mov %l0,%o1
	st %o1,[%o0]
	st %g0,[%l1]
	ld [%fp+72],%o0
	st %o0,[%l0+8]
	ld [%fp-28],%o0
	st %o0,[%l0]
	mov %l0,%o0
	mov %l0,%o1
	call _balance_case_nodes,0
	nop
	add %l0,4,%o1
	mov %o1,%o0
	mov %l0,%o1
	call _balance_case_nodes,0
	nop
	b L475
	nop
L468:
	ld [%fp+68],%o0
	ld [%o0],%l0
	ld [%fp+72],%o0
	st %o0,[%l0+8]
L476:
	ld [%l0+4],%o0
	cmp %o0,0
	be L477
	nop
	ld [%l0+4],%o0
	st %l0,[%o0+8]
L478:
	ld [%l0+4],%l0
	b L476
	nop
L477:
L475:
L464:
L463:
	ret
	restore
	.align 4
	.proc	04
_node_has_low_bound:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L480
	nop
	ld [%fp+68],%l0
	mov 1,%o0
	mov 0,%o1
	call _build_int_2,0
	nop
	mov %o0,%o2
	mov 64,%o0
	ld [%l0+12],%o1
	call _combine,0
	nop
	st %o0,[%fp-20]
	ld [%fp+68],%o1
	ld [%fp-20],%o0
	ld [%o1+12],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L481
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L482:
	ld [%fp-24],%o0
	cmp %o0,0
	be L483
	nop
	ld [%fp-24],%o1
	ld [%fp-20],%o0
	ld [%o1+16],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	be L485
	nop
	mov 1,%i0
	b L479
	nop
L485:
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L486
	nop
	b L483
	nop
L486:
L484:
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
	b L482
	nop
L483:
L481:
L480:
	mov 0,%i0
	b L479
	nop
L479:
	ret
	restore
	.align 4
	.proc	04
_node_has_high_bound:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	bne L488
	nop
	ld [%fp+68],%l0
	mov 1,%o0
	mov 0,%o1
	call _build_int_2,0
	nop
	mov %o0,%o2
	mov 63,%o0
	ld [%l0+16],%o1
	call _combine,0
	nop
	st %o0,[%fp-20]
	ld [%fp+68],%o1
	ld [%o1+16],%o0
	ld [%fp-20],%o1
	call _tree_int_cst_lt,0
	nop
	cmp %o0,0
	be L489
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L490:
	ld [%fp-24],%o0
	cmp %o0,0
	be L491
	nop
	ld [%fp-24],%o1
	ld [%fp-20],%o0
	ld [%o1+12],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	be L493
	nop
	mov 1,%i0
	b L487
	nop
L493:
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L494
	nop
	b L491
	nop
L494:
L492:
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
	b L490
	nop
L491:
L489:
L488:
	mov 0,%i0
	b L487
	nop
L487:
	ret
	restore
	.align 4
	.proc	04
_node_is_bounded:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0],%o1
	cmp %o1,0
	bne L497
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	bne L497
	nop
	b L496
	nop
L497:
	mov 0,%i0
	b L495
	nop
L496:
	mov 0,%l0
	ld [%fp+68],%o0
	call _node_has_low_bound,0
	nop
	cmp %o0,0
	be L498
	nop
	ld [%fp+68],%o0
	call _node_has_high_bound,0
	nop
	cmp %o0,0
	be L498
	nop
	mov 1,%l0
L498:
	mov %l0,%i0
	b L495
	nop
L495:
	ret
	restore
	.align 4
	.proc	020
_emit_jump_if_reachable:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	call _get_last_insn,0
	nop
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,16
	be L500
	nop
	ld [%fp+68],%o0
	call _emit_jump,0
	nop
L500:
L499:
	ret
	restore
	.align 4
	.proc	020
_emit_case_nodes:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	st %i3,[%fp+80]
	ld [%fp+80],%o0
	cmp %o0,0
	be L502
	nop
	sethi %hi(_gen_bgtu),%o1
	or %o1,%lo(_gen_bgtu),%o0
	b L503
	nop
L502:
	sethi %hi(_gen_bgt),%o1
	or %o1,%lo(_gen_bgt),%o0
L503:
	st %o0,[%fp-20]
	ld [%fp+80],%o0
	cmp %o0,0
	be L504
	nop
	sethi %hi(_gen_bgeu),%o1
	or %o1,%lo(_gen_bgeu),%o0
	b L505
	nop
L504:
	sethi %hi(_gen_bge),%o1
	or %o1,%lo(_gen_bge),%o0
L505:
	st %o0,[%fp-24]
	ld [%fp+80],%o0
	cmp %o0,0
	be L506
	nop
	sethi %hi(_gen_bltu),%o1
	or %o1,%lo(_gen_bltu),%o0
	b L507
	nop
L506:
	sethi %hi(_gen_blt),%o1
	or %o1,%lo(_gen_blt),%o0
L507:
	st %o0,[%fp-28]
	ld [%fp+80],%o0
	cmp %o0,0
	be L508
	nop
	sethi %hi(_gen_bleu),%o1
	or %o1,%lo(_gen_bleu),%o0
	b L509
	nop
L508:
	sethi %hi(_gen_ble),%o1
	or %o1,%lo(_gen_ble),%o0
L509:
	st %o0,[%fp-32]
	ld [%fp+72],%o0
	ld [%o0+20],%o1
	cmp %o1,0
	be L510
	nop
	ld [%fp+76],%o0
	call _emit_jump_if_reachable,0
	nop
	ld [%fp+72],%o1
	ld [%o1+20],%o0
	call _expand_label,0
	nop
L510:
	ld [%fp+72],%o0
	ld [%fp+72],%o1
	ld [%o0+12],%o0
	ld [%o1+16],%o1
	call _tree_int_cst_equal,0
	nop
	cmp %o0,0
	be L511
	nop
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%l0
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o2
	ld [%fp+68],%o0
	mov %l0,%o1
	ld [%fp+80],%o3
	call _do_jump_if_equal,0
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L512
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L513
	nop
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L514
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L515
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L516
	nop
L515:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L516:
	b L517
	nop
L514:
	ld [%fp+72],%o1
	ld [%o1],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L518
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-28],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	b L519
	nop
L518:
	mov 40,%o0
	mov 0,%o1
	mov 0,%o2
	call _build_decl,0
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o2
	st %o0,[%o2+20]
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+20],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L519:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L517:
	b L520
	nop
L513:
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+4],%o0
	cmp %o0,0
	be L521
	nop
	ld [%fp+72],%o0
	call _node_has_low_bound,0
	nop
	cmp %o0,0
	bne L521
	nop
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+76],%o0
	ld [%fp-28],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L521:
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L522
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L523
	nop
L522:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L523:
L520:
	b L524
	nop
L512:
	ld [%fp+72],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L525
	nop
	ld [%fp+72],%o1
	ld [%o1],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L526
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L527
	nop
L526:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L527:
L525:
L524:
	b L528
	nop
L511:
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	cmp %o1,0
	be L529
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L530
	nop
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L531
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	b L532
	nop
L531:
	mov 40,%o0
	mov 0,%o1
	mov 0,%o2
	call _build_decl,0
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o2
	st %o0,[%o2+20]
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+20],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L532:
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-24],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L533
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L534
	nop
L533:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L534:
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+20],%o0
	cmp %o0,0
	be L535
	nop
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L535:
	b L536
	nop
L530:
	ld [%fp+72],%o0
	call _node_has_low_bound,0
	nop
	cmp %o0,0
	bne L537
	nop
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+76],%o0
	ld [%fp-28],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L537:
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-32],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+4],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L538
	nop
	ld [%fp+72],%o0
	ld [%o0+4],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L539
	nop
L538:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L539:
L536:
	b L540
	nop
L529:
	ld [%fp+72],%o0
	ld [%o0],%o1
	cmp %o1,0
	be L541
	nop
	ld [%fp+72],%o0
	call _node_has_high_bound,0
	nop
	cmp %o0,0
	bne L542
	nop
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+76],%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L542:
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-24],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1],%o0
	call _node_is_bounded,0
	nop
	cmp %o0,0
	be L543
	nop
	ld [%fp+72],%o0
	ld [%o0],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump,0
	nop
	b L544
	nop
L543:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1],%o1
	ld [%fp+76],%o2
	ld [%fp+80],%o3
	call _emit_case_nodes,0
	nop
L544:
	b L545
	nop
L541:
	ld [%fp+72],%o0
	call _node_has_high_bound,0
	nop
	cmp %o0,0
	bne L546
	nop
	ld [%fp+72],%o1
	ld [%o1+16],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+76],%o0
	ld [%fp-20],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L546:
	ld [%fp+72],%o0
	call _node_has_low_bound,0
	nop
	cmp %o0,0
	bne L547
	nop
	ld [%fp+72],%o1
	ld [%o1+12],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	ld [%fp+68],%o0
	mov 0,%o2
	mov 0,%o3
	mov 0,%o4
	call _emit_cmp_insn,0
	nop
	ld [%fp+72],%o1
	ld [%o1+24],%o0
	call _label_rtx,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-24],%o1
	call %o1,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
L547:
L545:
L540:
L528:
L501:
	ret
	restore
	.align 4
	.global _get_frame_size
	.proc	04
_get_frame_size:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_frame_offset),%o0
	ld [%o0+%lo(_frame_offset)],%o1
	sub %g0,%o1,%o0
	mov %o0,%i0
	b L548
	nop
L548:
	ret
	restore
	.align 4
	.global _assign_stack_local
	.proc	0110
_assign_stack_local:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %g0,[%fp-20]
	sethi %hi(_frame_pointer_needed),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_frame_pointer_needed)]
	ld [%fp+72],%o0
	add %o0,1,%o1
	mov %o1,%o0
	srl %o0,31,%o1
	add %o0,%o1,%o0
	sra %o0,1,%o1
	mov %o1,%o0
	sll %o0,1,%o1
	st %o1,[%fp+72]
	ld [%fp+68],%o0
	cmp %o0,26
	be L550
	nop
	ld [%fp+68],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%fp+72],%o2
	ld [%o0+%o1],%o0
	sub %o2,%o0,%o1
	st %o1,[%fp-20]
L550:
	sethi %hi(_frame_offset),%o0
	sethi %hi(_frame_offset),%o1
	ld [%o1+%lo(_frame_offset)],%o2
	ld [%fp+72],%o1
	sub %o2,%o1,%o2
	st %o2,[%o0+%lo(_frame_offset)]
	sethi %hi(_frame_pointer_rtx),%l2
	sethi %hi(_frame_offset),%o0
	ld [%o0+%lo(_frame_offset)],%o1
	ld [%fp-20],%o0
	add %o1,%o0,%o2
	mov 30,%o0
	mov 0,%o1
	call _gen_rtx,0
	nop
	mov %o0,%o3
	mov 44,%o0
	mov 4,%o1
	ld [%l2+%lo(_frame_pointer_rtx)],%o2
	call _gen_rtx,0
	nop
	mov %o0,%l1
	ld [%fp+68],%o0
	mov %l1,%o1
	call _memory_address_p,0
	nop
	cmp %o0,0
	bne L551
	nop
	sethi %hi(_invalid_stack_slot),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_invalid_stack_slot)]
L551:
	mov 37,%o0
	ld [%fp+68],%o1
	mov %l1,%o2
	call _gen_rtx,0
	nop
	mov %o0,%l0
	sethi %hi(_stack_slot_list),%o3
	mov 2,%o0
	mov 0,%o1
	mov %l0,%o2
	ld [%o3+%lo(_stack_slot_list)],%o3
	call _gen_rtx,0
	nop
	sethi %hi(_stack_slot_list),%o1
	st %o0,[%o1+%lo(_stack_slot_list)]
	mov %l0,%i0
	b L549
	nop
L549:
	ret
	restore
	.align 4
	.global _put_var_into_stack
	.proc	020
_put_var_into_stack:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+64],%l0
	cmp %l0,0
	bne L553
	nop
	b L552
	nop
L553:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	be L554
	nop
	b L552
	nop
L554:
	mov %l0,%o0
	call _parm_stack_loc,0
	nop
	mov %o0,%l1
	cmp %l1,0
	bne L555
	nop
	ld [%l0],%o0
	srl %o0,8,%o1
	and %o1,255,%o0
	ld [%l0],%o2
	srl %o2,8,%o1
	and %o1,255,%o2
	mov %o2,%o3
	sll %o3,2,%o1
	sethi %hi(_mode_size),%o3
	or %o3,%lo(_mode_size),%o2
	ld [%o1+%o2],%o1
	call _assign_stack_local,0
	nop
	mov %o0,%l1
L555:
	ld [%l1+4],%o0
	st %o0,[%l0+4]
	ld [%l0],%o0
	and %o0,-17,%o1
	st %o1,[%l0]
	mov 37,%o0
	sth %o0,[%l0]
	mov 0,%o0
	ld [%fp+68],%o1
	ld [%o1+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o1
	cmp %o1,16
	be L557
	nop
	ld [%fp+68],%o1
	ld [%o1+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o1
	cmp %o1,19
	be L557
	nop
	ld [%fp+68],%o1
	ld [%o1+8],%o2
	ldub [%o2+12],%o3
	and %o3,0xff,%o1
	cmp %o1,20
	be L557
	nop
	b L556
	nop
L557:
	mov 1,%o0
L556:
	and %o0,1,%o1
	sll %o1,3,%o0
	ld [%l0],%o2
	and %o2,-9,%o1
	or %o1,%o0,%o1
	st %o1,[%l0]
	mov %l0,%o0
	call _fixup_var_refs,0
	nop
L552:
	ret
	restore
	.align 4
	.proc	020
_fixup_var_refs:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_sequence_stack),%o0
	ld [%o0+%lo(_sequence_stack)],%o1
	st %o1,[%fp-20]
	sethi %hi(_sequence_stack),%o0
	ld [%o0+%lo(_sequence_stack)],%o1
	st %o1,[%fp-20]
	call _get_insns,0
	nop
	mov %o0,%o1
	ld [%fp-20],%o0
	xor %o0,0,%o3
	subcc %g0,%o3,%g0
	subx %g0,-1,%o2
	ld [%fp+68],%o0
	call _fixup_var_refs_insns,0
	nop
L559:
	ld [%fp-20],%o0
	cmp %o0,0
	be L560
	nop
	ld [%fp-20],%o1
	ld [%o1+4],%o0
	call _push_to_sequence,0
	nop
	ld [%fp-20],%o1
	ld [%fp-20],%o0
	ld [%o0+8],%o2
	ld [%o2+8],%o0
	xor %o0,0,%o3
	subcc %g0,%o3,%g0
	subx %g0,-1,%o2
	ld [%fp+68],%o0
	ld [%o1+4],%o1
	call _fixup_var_refs_insns,0
	nop
	call _end_sequence,0
	nop
L561:
	ld [%fp-20],%o0
	ld [%o0+8],%o1
	ld [%o1+8],%o0
	st %o0,[%fp-20]
	b L559
	nop
L560:
	nop
	sethi %hi(_rtl_expr_chain),%o0
	ld [%o0+%lo(_rtl_expr_chain)],%o1
	st %o1,[%fp-24]
L562:
	ld [%fp-24],%o0
	cmp %o0,0
	be L563
	nop
	ld [%fp-24],%o0
	mov 20,%o1
	ld [%o0+20],%o2
	add %o1,%o2,%o0
	ld [%o0],%o1
	st %o1,[%fp-28]
	sethi %hi(_const0_rtx),%o0
	ld [%fp-28],%o1
	ld [%o0+%lo(_const0_rtx)],%o0
	cmp %o1,%o0
	be L565
	nop
	ld [%fp-28],%o0
	cmp %o0,0
	be L565
	nop
	ld [%fp-28],%o0
	call _push_to_sequence,0
	nop
	ld [%fp+68],%o0
	ld [%fp-28],%o1
	mov 0,%o2
	call _fixup_var_refs_insns,0
	nop
	call _end_sequence,0
	nop
L565:
L564:
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	b L562
	nop
L563:
L558:
	ret
	restore
	.align 4
	.proc	020
_fixup_var_refs_insns:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
L567:
	ld [%fp+72],%o0
	cmp %o0,0
	be L568
	nop
	ld [%fp+72],%o0
	ld [%o0+12],%o1
	st %o1,[%fp-20]
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,13
	be L570
	nop
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,15
	be L570
	nop
	ld [%fp+72],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,14
	be L570
	nop
	b L569
	nop
L570:
	ld [%fp+76],%o0
	cmp %o0,0
	be L571
	nop
	ld [%fp+72],%o0
	ld [%o0+16],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,25
	bne L571
	nop
	ld [%fp+72],%o0
	ld [%o0+16],%o1
	ld [%o1+4],%o0
	ld [%fp+68],%o1
	cmp %o0,%o1
	bne L571
	nop
	ld [%fp+72],%o0
	ld [%o0+16],%o1
	ld [%o1+8],%o0
	ld [%fp+68],%o1
	call _rtx_equal_p,0
	nop
	cmp %o0,0
	be L571
	nop
	ld [%fp+72],%o0
	call _delete_insn,0
	nop
	st %o0,[%fp-20]
	sethi %hi(_last_parm_insn),%o0
	ld [%fp+72],%o1
	ld [%o0+%lo(_last_parm_insn)],%o0
	cmp %o1,%o0
	bne L572
	nop
	sethi %hi(_last_parm_insn),%o0
	ld [%fp-20],%o1
	ld [%o1+8],%o2
	st %o2,[%o0+%lo(_last_parm_insn)]
L572:
	b L573
	nop
L571:
	ld [%fp+72],%o1
	ld [%fp+68],%o0
	ld [%o1+16],%o1
	ld [%fp+72],%o2
	call _fixup_var_refs_1,0
	nop
L573:
	nop
	ld [%fp+72],%o0
	ld [%o0+28],%o1
	st %o1,[%fp-24]
L574:
	ld [%fp-24],%o0
	cmp %o0,0
	be L575
	nop
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,3
	be L577
	nop
	ld [%fp-24],%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	call _walk_fixup_memory_subreg,0
	nop
	ld [%fp-24],%o1
	st %o0,[%o1+4]
L577:
L576:
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
	b L574
	nop
L575:
L569:
	ld [%fp-20],%o0
	st %o0,[%fp+72]
	b L567
	nop
L568:
L566:
	ret
	restore
	.align 4
	.proc	0110
_fixup_var_refs_1:
	!#PROLOGUE# 0
	save %sp,-144,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	mov %i1,%l1
	st %i2,[%fp+76]
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	st %o0,[%fp-20]
	ld [%fp-20],%o1
	add %o1,-25,%o0
	mov 65,%o1
	cmp %o1,%o0
	blu L628
	nop
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(L627),%o2
	or %o2,%lo(L627),%o1
	ld [%o0+%o1],%o0
	jmp %o0
	nop
L627:
	.word	L598
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L585
	.word	L585
	.word	L585
	.word	L585
	.word	L585
	.word	L592
	.word	L628
	.word	L580
	.word	L588
	.word	L588
	.word	L588
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L628
	.word	L590
	.word	L590
L580:
	cmp %l0,%l1
	bne L581
	nop
	mov %l1,%o0
	ld [%fp+76],%o1
	call _fixup_stack_1,0
	nop
	mov %o0,%l1
	ld [%l1],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	mov %o0,%l4
	mov %l4,%o0
	mov %l1,%o1
	call _gen_move_insn,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call _emit_insn_before,0
	nop
	mov %l4,%i0
	b L578
	nop
L581:
	b L579
	nop
L582:
L583:
L584:
L585:
L586:
L587:
L588:
L589:
	mov %l1,%i0
	b L578
	nop
L590:
L591:
L592:
	mov %l1,%l4
L593:
	lduh [%l4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,35
	be L595
	nop
	lduh [%l4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,89
	be L595
	nop
	lduh [%l4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,90
	be L595
	nop
	b L594
	nop
L595:
	ld [%l4+4],%l4
	b L593
	nop
L594:
	cmp %l4,%l0
	bne L596
	nop
	mov %l1,%o0
	ld [%fp+76],%o1
	call _fixup_stack_1,0
	nop
	mov %o0,%l1
	ld [%l1],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	mov %o0,%l4
	lduh [%l1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,35
	bne L597
	nop
	mov %l1,%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	mov %o0,%l1
L597:
	mov %l4,%o0
	mov %l1,%o1
	call _gen_move_insn,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call _emit_insn_before,0
	nop
	mov %l4,%i0
	b L578
	nop
L596:
	b L579
	nop
L598:
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L600
	nop
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L600
	nop
	b L599
	nop
L600:
	mov %l1,%o0
	ld [%fp+76],%o1
	mov 0,%o2
	call _optimize_bit_field,0
	nop
L599:
	ld [%l1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L602
	nop
	ld [%l1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L602
	nop
	b L601
	nop
L602:
	mov %l1,%o0
	ld [%fp+76],%o1
	mov 0,%o2
	call _optimize_bit_field,0
	nop
L601:
	ld [%l1+4],%o0
	st %o0,[%fp-24]
	ld [%l1+8],%o0
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	st %o0,[%fp-32]
	ld [%fp-28],%o0
	st %o0,[%fp-36]
L603:
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	be L605
	nop
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,36
	be L605
	nop
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L605
	nop
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L605
	nop
	b L604
	nop
L605:
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	b L603
	nop
L604:
	nop
L606:
	ld [%fp-28],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	be L608
	nop
	ld [%fp-28],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L608
	nop
	ld [%fp-28],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L608
	nop
	b L607
	nop
L608:
	ld [%fp-28],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-28]
	b L606
	nop
L607:
	ld [%fp-28],%o0
	cmp %o0,%l0
	be L609
	nop
	ld [%fp-24],%o0
	cmp %o0,%l0
	be L609
	nop
	b L579
	nop
L609:
	ld [%fp-32],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L611
	nop
	ld [%fp-32],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L611
	nop
	b L610
	nop
L611:
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,35
	bne L610
	nop
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	ld [%o1+4],%o0
	cmp %o0,%l0
	bne L610
	nop
	ld [%fp-32],%o1
	ld [%o1+4],%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	ld [%fp-32],%o1
	st %o0,[%o1+4]
L610:
	ld [%fp-36],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L613
	nop
	ld [%fp-36],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L613
	nop
	b L612
	nop
L613:
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,35
	bne L612
	nop
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	ld [%o1+4],%o0
	cmp %o0,%l0
	bne L612
	nop
	ld [%fp-36],%o1
	ld [%o1+4],%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	ld [%fp-36],%o1
	st %o0,[%o1+4]
L612:
	ld [%fp-32],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L615
	nop
	ld [%fp-32],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L615
	nop
	b L614
	nop
L615:
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,37
	bne L614
	nop
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	ldub [%o1+2],%o2
	and %o2,0xff,%o0
	cmp %o0,1
	be L614
	nop
	ld [%fp-32],%o1
	ld [%o1+4],%o0
	call _copy_rtx,0
	nop
	ld [%fp-32],%o1
	st %o0,[%o1+4]
	ld [%fp-32],%o0
	ld [%o0+4],%o1
	mov 1,%o0
	stb %o0,[%o1+2]
L614:
	ld [%fp-36],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,89
	be L617
	nop
	ld [%fp-36],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,90
	be L617
	nop
	b L616
	nop
L617:
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,37
	bne L616
	nop
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	ldub [%o1+2],%o2
	and %o2,0xff,%o0
	cmp %o0,1
	be L616
	nop
	ld [%fp-36],%o1
	ld [%o1+4],%o0
	call _copy_rtx,0
	nop
	ld [%fp-36],%o1
	st %o0,[%o1+4]
	ld [%fp-36],%o0
	ld [%o0+4],%o1
	mov 1,%o0
	stb %o0,[%o1+2]
L616:
	ld [%fp-24],%o0
	cmp %o0,%l0
	bne L618
	nop
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,36
	bne L618
	nop
	ld [%l1+4],%o0
	ld [%o0+4],%o1
	st %o1,[%l1+4]
L618:
	ld [%l1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	be L620
	nop
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	be L620
	nop
	ld [%l1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L621
	nop
	ld [%l1+8],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	be L620
	nop
	b L621
	nop
L621:
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L619
	nop
	ld [%l1+4],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	be L620
	nop
	b L619
	nop
L620:
	ld [%fp-28],%o0
	cmp %o0,%l0
	bne L622
	nop
	ld [%l1+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L622
	nop
	ld [%l1+8],%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	st %o0,[%l1+8]
L622:
	ld [%fp-24],%o0
	cmp %o0,%l0
	bne L623
	nop
	ld [%l1+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L623
	nop
	ld [%l1+4],%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	st %o0,[%l1+4]
L623:
	mov %l1,%o0
	ld [%fp+76],%o1
	call _fixup_stack_1,0
	nop
	mov %o0,%i0
	b L578
	nop
L619:
	ld [%fp-24],%o0
	cmp %o0,%l0
	bne L624
	nop
	ld [%l1+4],%l4
	lduh [%l4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,36
	bne L625
	nop
	ld [%l4+4],%l4
L625:
	lduh [%l4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,35
	bne L626
	nop
	mov %l4,%o0
	ld [%fp+76],%o1
	call _fixup_memory_subreg,0
	nop
	mov %o0,%l4
L626:
	mov %l4,%o0
	ld [%fp+76],%o1
	call _fixup_stack_1,0
	nop
	st %o0,[%fp-44]
	ld [%l4],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%fp-40]
	ld [%fp-44],%o0
	ld [%fp-40],%o1
	call _gen_move_insn,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+76],%o1
	call _emit_insn_after,0
	nop
	ld [%fp-40],%o0
	st %o0,[%l1+4]
L624:
L628:
L579:
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l3
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l2
L629:
	cmp %l2,0
	bl L630
	nop
	ldub [%l3+%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L632
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	add %l1,%o1,%o2
	mov %l0,%o0
	ld [%o2+4],%o1
	ld [%fp+76],%o2
	call _fixup_var_refs_1,0
	nop
	mov %l2,%o1
	sll %o1,2,%o2
	add %l1,%o2,%o1
	st %o0,[%o1+4]
L632:
	ldub [%l3+%l2],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L633
	nop
	mov 0,%l5
L634:
	mov %l2,%o0
	sll %o0,2,%o1
	add %l1,%o1,%o2
	ld [%o2+4],%o0
	ld [%o0],%o1
	cmp %l5,%o1
	bgeu L635
	nop
	mov %l2,%o0
	sll %o0,2,%o1
	add %l1,%o1,%o2
	ld [%o2+4],%o0
	mov %l5,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	mov %l0,%o0
	ld [%o1+4],%o1
	ld [%fp+76],%o2
	call _fixup_var_refs_1,0
	nop
	mov %l2,%o1
	sll %o1,2,%o2
	add %l1,%o2,%o3
	ld [%o3+4],%o1
	mov %l5,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L636:
	add %l5,1,%l5
	b L634
	nop
L635:
L633:
L631:
	add %l2,-1,%l2
	b L629
	nop
L630:
	mov %l1,%i0
	b L578
	nop
L578:
	ret
	restore
	.align 4
	.proc	0110
_fixup_memory_subreg:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	mov %o1,%o0
	sll %o0,2,%o1
	st %o1,[%fp-20]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	ld [%o1+4],%o0
	st %o0,[%fp-24]
	ld [%fp+68],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	st %o1,[%fp-28]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	ld [%o1],%o2
	srl %o2,8,%o0
	and %o0,255,%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	cmp %o0,4
	ble L638
	nop
	mov 4,%o0
L638:
	ld [%fp-28],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	sethi %hi(_mode_size),%o3
	or %o3,%lo(_mode_size),%o2
	ld [%o1+%o2],%o1
	cmp %o1,4
	ble L639
	nop
	mov 4,%o1
L639:
	sub %o0,%o1,%o0
	ld [%fp-20],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-20]
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	call _plus_constant,0
	nop
	st %o0,[%fp-24]
	ld [%fp-28],%o0
	ld [%fp-24],%o1
	call _memory_address_p,0
	nop
	cmp %o0,0
	be L640
	nop
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp-28],%o1
	ld [%fp-24],%o2
	call _change_address,0
	nop
	mov %o0,%i0
	b L637
	nop
L640:
	call _start_sequence,0
	nop
	st %o0,[%fp-32]
	ld [%fp+68],%o1
	ld [%o1+4],%o0
	ld [%fp-28],%o1
	ld [%fp-24],%o2
	call _change_address,0
	nop
	st %o0,[%fp-36]
	call _gen_sequence,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+72],%o1
	call _emit_insn_before,0
	nop
	ld [%fp-32],%o0
	call _end_sequence,0
	nop
	ld [%fp-36],%i0
	b L637
	nop
L637:
	ret
	restore
	.align 4
	.proc	0110
_walk_fixup_memory_subreg:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	mov %i0,%l0
	st %i1,[%fp+72]
	cmp %l0,0
	bne L642
	nop
	mov 0,%i0
	b L641
	nop
L642:
	lduh [%l0],%o0
	sll %o0,16,%o1
	srl %o1,16,%l1
	cmp %l1,35
	bne L643
	nop
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L643
	nop
	mov %l0,%o0
	ld [%fp+72],%o1
	call _fixup_memory_subreg,0
	nop
	mov %o0,%i0
	b L641
	nop
L643:
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l2
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l3
L644:
	cmp %l3,0
	bl L645
	nop
	ldub [%l2+%l3],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L647
	nop
	mov %l3,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	ld [%fp+72],%o1
	call _walk_fixup_memory_subreg,0
	nop
	mov %l3,%o1
	sll %o1,2,%o2
	add %l0,%o2,%o1
	st %o0,[%o1+4]
L647:
	ldub [%l2+%l3],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L648
	nop
	mov 0,%l4
L649:
	mov %l3,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	ld [%o0],%o1
	cmp %l4,%o1
	bgeu L650
	nop
	mov %l3,%o0
	sll %o0,2,%o1
	add %l0,%o1,%o2
	ld [%o2+4],%o0
	mov %l4,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	call _walk_fixup_memory_subreg,0
	nop
	mov %l3,%o1
	sll %o1,2,%o2
	add %l0,%o2,%o3
	ld [%o3+4],%o1
	mov %l4,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L651:
	add %l4,1,%l4
	b L649
	nop
L650:
L648:
L646:
	add %l3,-1,%l3
	b L644
	nop
L645:
	mov %l0,%i0
	b L641
	nop
L641:
	ret
	restore
	.align 4
	.proc	0110
_fixup_stack_1:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o0
	srl %o0,16,%l1
	cmp %l1,37
	bne L653
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%l3
	lduh [%l3],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,44
	bne L654
	nop
	sethi %hi(_frame_pointer_rtx),%o0
	ld [%l3+4],%o1
	ld [%o0+%lo(_frame_pointer_rtx)],%o0
	cmp %o1,%o0
	bne L654
	nop
	ld [%l3+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L654
	nop
	ld [%fp+68],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	mov %l3,%o1
	call _memory_address_p,0
	nop
	cmp %o0,0
	be L655
	nop
	ld [%fp+68],%i0
	b L652
	nop
L655:
	ld [%l3],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%fp-20]
	ld [%fp-20],%o0
	mov %l3,%o1
	call _gen_move_insn,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp+72],%o1
	call _emit_insn_before,0
	nop
	ld [%fp+68],%o0
	mov 0,%o1
	ld [%fp-20],%o2
	call _change_address,0
	nop
	mov %o0,%i0
	b L652
	nop
L654:
	ld [%fp+68],%i0
	b L652
	nop
L653:
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_format),%o2
	or %o2,%lo(_rtx_format),%o1
	ld [%o0+%o1],%l2
	mov %l1,%o1
	sll %o1,2,%o0
	sethi %hi(_rtx_length),%o2
	or %o2,%lo(_rtx_length),%o1
	ld [%o0+%o1],%o0
	add %o0,-1,%l0
L656:
	cmp %l0,0
	bl L657
	nop
	ldub [%l2+%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,101
	bne L659
	nop
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	call _fixup_stack_1,0
	nop
	ld [%fp+68],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L659:
	ldub [%l2+%l0],%o0
	sll %o0,24,%o1
	sra %o1,24,%o0
	cmp %o0,69
	bne L660
	nop
	mov 0,%l3
L661:
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%o0],%o1
	cmp %l3,%o1
	bgeu L662
	nop
	ld [%fp+68],%o0
	mov %l0,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	mov %l3,%o1
	sll %o1,2,%o2
	add %o0,%o2,%o1
	ld [%o1+4],%o0
	ld [%fp+72],%o1
	call _fixup_stack_1,0
	nop
	ld [%fp+68],%o1
	mov %l0,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o2
	ld [%o2+4],%o1
	mov %l3,%o2
	sll %o2,2,%o3
	add %o1,%o3,%o1
	st %o0,[%o1+4]
L663:
	add %l3,1,%l3
	b L661
	nop
L662:
L660:
L658:
	add %l0,-1,%l0
	b L656
	nop
L657:
	ld [%fp+68],%i0
	b L652
	nop
L652:
	ret
	restore
	.align 4
	.proc	020
_optimize_bit_field:
	!#PROLOGUE# 0
	save %sp,-128,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
	st %i2,[%fp+76]
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,89
	be L666
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,90
	be L666
	nop
	b L665
	nop
L666:
	ld [%fp+68],%o0
	ld [%o0+4],%l0
	mov 1,%o0
	st %o0,[%fp-20]
	b L667
	nop
L665:
	ld [%fp+68],%o0
	ld [%o0+8],%l0
	st %g0,[%fp-20]
L667:
	ld [%l0+8],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L668
	nop
	ld [%l0+12],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L668
	nop
	ld [%l0+8],%o0
	sethi %hi(_mode_size+4),%o1
	ld [%o1+%lo(_mode_size+4)],%o2
	mov %o2,%o1
	sll %o1,3,%o2
	ld [%o0+4],%o0
	cmp %o0,%o2
	be L669
	nop
	ld [%l0+8],%o0
	sethi %hi(_mode_size+8),%o1
	ld [%o1+%lo(_mode_size+8)],%o2
	mov %o2,%o1
	sll %o1,3,%o2
	ld [%o0+4],%o0
	cmp %o0,%o2
	be L669
	nop
	b L668
	nop
L669:
	ld [%l0+12],%o0
	ld [%l0+8],%o1
	ld [%o0+4],%o0
	ld [%o1+4],%o1
	call .rem,0
	nop
	cmp %o0,0
	bne L668
	nop
	mov 0,%l1
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L670
	nop
	ld [%l0+4],%l1
	b L671
	nop
L670:
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L672
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L672
	nop
	ld [%l0+4],%o0
	ld [%o0+4],%o1
	mov %o1,%o0
	sll %o0,2,%o1
	ld [%fp+76],%o2
	add %o1,%o2,%o0
	ld [%o0],%l1
	b L673
	nop
L672:
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L674
	nop
	ld [%l0+4],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,37
	bne L674
	nop
	ld [%l0+4],%o0
	ld [%o0+4],%l1
	b L675
	nop
L674:
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L676
	nop
	ld [%fp+76],%o0
	cmp %o0,0
	be L676
	nop
	ld [%l0+4],%o0
	ld [%o0+4],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L676
	nop
	ld [%l0+4],%o0
	ld [%o0+4],%o1
	ld [%o1+4],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	ld [%fp+76],%o1
	add %o0,%o1,%o0
	ld [%o0],%l1
L676:
L675:
L673:
L671:
	cmp %l1,0
	be L677
	nop
	ld [%l1+4],%o0
	call _mode_dependent_address_p,0
	nop
	cmp %o0,0
	bne L677
	nop
	ld [%l0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov 0,%o0
	ld [%l1+4],%o2
	call _offsetable_address_p,0
	nop
	cmp %o0,0
	be L677
	nop
	ld [%l0+12],%o0
	sethi %hi(_mode_size+4),%o1
	ld [%o1+%lo(_mode_size+4)],%o2
	mov %o2,%o1
	sll %o1,3,%o2
	ld [%o0+4],%o0
	mov %o2,%o1
	call .div,0
	nop
	mov %o0,%l2
	ld [%l0+4],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L678
	nop
	ld [%l0+4],%o0
	ld [%o0+8],%o1
	mov %o1,%o0
	sll %o0,2,%o1
	add %l2,%o1,%l2
	ld [%l0+4],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	cmp %o0,4
	ble L679
	nop
	mov 4,%o0
L679:
	ld [%l1],%o2
	srl %o2,8,%o1
	and %o1,255,%o2
	mov %o2,%o3
	sll %o3,2,%o1
	sethi %hi(_mode_size),%o3
	or %o3,%lo(_mode_size),%o2
	ld [%o1+%o2],%o1
	cmp %o1,4
	ble L680
	nop
	mov 4,%o1
L680:
	sub %o0,%o1,%o0
	sub %l2,%o0,%l2
L678:
	ld [%l0+8],%o0
	sethi %hi(_mode_size+4),%o1
	ld [%o1+%lo(_mode_size+4)],%o2
	mov %o2,%o1
	sll %o1,3,%o2
	ld [%o0+4],%o0
	cmp %o0,%o2
	bne L681
	nop
	mov 1,%o1
	b L682
	nop
L681:
	mov 2,%o1
L682:
	mov 37,%o0
	ld [%l1+4],%o2
	call _gen_rtx,0
	nop
	mov %o0,%l1
	ld [%fp-20],%o0
	cmp %o0,0
	be L683
	nop
	mov %l1,%o0
	mov %l2,%o1
	call _adj_offsetable_operand,0
	nop
	ld [%fp+68],%o1
	st %o0,[%o1+4]
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,38
	be L684
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,39
	be L684
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,30
	be L684
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,32
	be L684
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-24]
L685:
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L686
	nop
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	bne L686
	nop
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	b L685
	nop
L686:
	ld [%fp-24],%o0
	ldub [%o0+2],%o1
	and %o1,0xff,%o0
	ldub [%l1+2],%o2
	and %o2,0xff,%o1
	cmp %o0,%o1
	be L687
	nop
	ld [%l1],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	ld [%fp+68],%o2
	mov %o1,%o0
	ld [%o2+8],%o1
	call _gen_lowpart,0
	nop
	st %o0,[%fp-24]
L687:
	ld [%fp+68],%o0
	ld [%fp-24],%o1
	st %o1,[%o0+8]
	b L688
	nop
L684:
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ldub [%o1+2],%o2
	and %o2,0xff,%o0
	cmp %o0,0
	be L689
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ldub [%o1+2],%o2
	and %o2,0xff,%o0
	ldub [%l1+2],%o2
	and %o2,0xff,%o1
	cmp %o0,%o1
	be L689
	nop
	call _abort,0
	nop
L689:
L688:
	b L690
	nop
L683:
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
L691:
	ld [%fp-24],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,35
	bne L692
	nop
	ld [%fp-24],%o0
	ld [%o0+8],%o1
	cmp %o1,0
	bne L692
	nop
	ld [%fp-24],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-24]
	b L691
	nop
L692:
	ld [%fp+68],%o0
	ld [%fp-24],%o1
	st %o1,[%o0+4]
	mov %l1,%o0
	mov %l2,%o1
	call _adj_offsetable_operand,0
	nop
	mov %o0,%l1
	ld [%fp-24],%o0
	ldub [%o0+2],%o1
	and %o1,0xff,%o0
	ldub [%l1+2],%o2
	and %o2,0xff,%o1
	cmp %o0,%o1
	bne L693
	nop
	ld [%fp+68],%o0
	st %l1,[%o0+8]
	b L694
	nop
L693:
	call _get_last_insn,0
	nop
	st %o0,[%fp-28]
	ld [%fp-24],%o0
	ld [%o0],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%fp-32]
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	lduh [%o1],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	xor %o0,90,%o1
	subcc %g0,%o1,%g0
	subx %g0,-1,%o2
	ld [%fp-32],%o0
	mov %l1,%o1
	call _convert_move,0
	nop
	ld [%fp-28],%l3
	call _get_last_insn,0
	nop
	mov %o0,%o1
	ld [%fp+72],%o2
	ld [%l3+12],%o0
	ld [%o2+8],%o2
	call _reorder_insns,0
	nop
	ld [%fp+68],%o0
	ld [%fp-32],%o1
	st %o1,[%o0+8]
L694:
L690:
	ld [%fp+72],%o0
	mov -1,%o1
	st %o1,[%o0+20]
L677:
L668:
L664:
	ret
	restore
	.align 4
	.global _max_parm_reg_num
	.proc	04
_max_parm_reg_num:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_max_parm_reg),%o0
	ld [%o0+%lo(_max_parm_reg)],%i0
	b L695
	nop
L695:
	ret
	restore
	.align 4
	.global _get_first_nonparm_insn
	.proc	0110
_get_first_nonparm_insn:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	sethi %hi(_last_parm_insn),%o0
	ld [%o0+%lo(_last_parm_insn)],%o1
	cmp %o1,0
	be L697
	nop
	sethi %hi(_last_parm_insn),%o1
	ld [%o1+%lo(_last_parm_insn)],%o0
	ld [%o0+12],%i0
	b L696
	nop
L697:
	call _get_insns,0
	nop
	mov %o0,%i0
	b L696
	nop
L696:
	ret
	restore
	.align 4
	.proc	0110
_parm_stack_loc:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	sethi %hi(_max_parm_reg),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_max_parm_reg)],%o1
	cmp %o0,%o1
	bge L699
	nop
	ld [%fp+68],%o0
	ld [%o0+4],%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_parm_reg_stack_loc),%o2
	ld [%o2+%lo(_parm_reg_stack_loc)],%o1
	add %o0,%o1,%o0
	ld [%o0],%i0
	b L698
	nop
L699:
	mov 0,%i0
	b L698
	nop
L698:
	ret
	restore
	.align 8
LC13:
	.ascii "__builtin_va_alist\0"
	.align 4
	.proc	020
_assign_parms:
	!#PROLOGUE# 0
	save %sp,-264,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	mov 8,%o0
	st %o0,[%fp-36]
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-40]
	ld [%fp+68],%o1
	ld [%o1+52],%o0
	call _list_length,0
	nop
	add %o0,56,%o1
	st %o1,[%fp-44]
	st %g0,[%fp-108]
	ld [%fp+68],%o0
	ld [%o0+52],%o1
	cmp %o1,0
	be L703
	nop
	ld [%fp+68],%o0
	ld [%o0+52],%o1
	ld [%o1+36],%o0
	cmp %o0,0
	be L703
	nop
	ld [%fp+68],%o0
	ld [%o0+52],%o1
	ld [%o1+36],%o2
	ld [%o2+20],%o0
	sethi %hi(LC13),%o2
	or %o2,%lo(LC13),%o1
	call _strcmp,0
	nop
	cmp %o0,0
	bne L703
	nop
	b L702
	nop
L703:
	ld [%fp-40],%o0
	ld [%o0+16],%o1
	cmp %o1,0
	be L701
	nop
	ld [%fp-40],%o1
	ld [%o1+16],%o0
	call _tree_last,0
	nop
	sethi %hi(_void_type_node),%o1
	ld [%o0+20],%o0
	ld [%o1+%lo(_void_type_node)],%o1
	cmp %o0,%o1
	bne L702
	nop
	b L701
	nop
L702:
	mov 1,%o5
	st %o5,[%fp-108]
L701:
	ld [%fp-108],%o5
	st %o5,[%fp-48]
	st %g0,[%fp-32]
	st %g0,[%fp-28]
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+28],%o0
	cmp %o0,26
	bne L704
	nop
	sethi %hi(_struct_value_incoming_rtx),%o1
	ld [%o1+%lo(_struct_value_incoming_rtx)],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L704
	nop
	sethi %hi(_mode_size+16),%o0
	ld [%fp-32],%o1
	ld [%o0+%lo(_mode_size+16)],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-32]
L704:
	sethi %hi(_parm_reg_stack_loc),%o5
	st %o5,[%fp-116]
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _oballoc,0
	nop
	ld [%fp-116],%o5
	st %o0,[%o5+%lo(_parm_reg_stack_loc)]
	sethi %hi(_parm_reg_stack_loc),%o0
	ld [%fp-44],%o1
	mov %o1,%o3
	sll %o3,2,%o2
	ld [%o0+%lo(_parm_reg_stack_loc)],%o0
	mov 0,%o1
	call _memset,0
	nop
	st %g0,[%fp-100]
	ld [%fp+68],%o0
	ld [%o0+52],%i3
L705:
	cmp %i3,0
	be L706
	nop
	mov 0,%o0
	ld [%i3+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o1
	cmp %o1,16
	be L709
	nop
	ld [%i3+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o1
	cmp %o1,19
	be L709
	nop
	ld [%i3+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o1
	cmp %o1,20
	be L709
	nop
	b L708
	nop
L709:
	mov 1,%o0
L708:
	st %o0,[%fp-52]
	mov -1,%o0
	st %o0,[%i3+44]
	sethi %hi(_error_mark_node),%o0
	ld [%i3+8],%o1
	ld [%o0+%lo(_error_mark_node)],%o0
	cmp %o1,%o0
	be L711
	nop
	ldub [%i3+12],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	bne L711
	nop
	ld [%i3+52],%o0
	cmp %o0,0
	bne L710
	nop
	b L711
	nop
L711:
	sethi %hi(_const0_rtx),%o2
	mov 37,%o0
	mov 26,%o1
	ld [%o2+%lo(_const0_rtx)],%o2
	call _gen_rtx,0
	nop
	st %o0,[%i3+64]
	ld [%i3+12],%o0
	or %o0,256,%o1
	st %o1,[%i3+12]
	b L707
	nop
L710:
	ld [%i3+52],%o0
	ldub [%o0+28],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-20]
	ld [%i3+8],%o0
	ldub [%o0+28],%o1
	and %o1,0xff,%o0
	st %o0,[%fp-24]
	ld [%fp-32],%o0
	st %o0,[%fp-64]
	ld [%fp-28],%o0
	st %o0,[%fp-60]
	ld [%fp-64],%o0
	ld [%fp-36],%o1
	add %o0,%o1,%o0
	st %o0,[%fp-64]
	ld [%fp-20],%o0
	cmp %o0,26
	bne L714
	nop
	ld [%i3+52],%o0
	call _size_in_bytes,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	bne L712
	nop
	ld [%i3+52],%o0
	call _size_in_bytes,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	ld [%o0+4],%l0
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L716
	nop
	cmp %l0,1
	ble L715
	nop
	b L712
	nop
L716:
	cmp %l0,3
	ble L715
	nop
	b L712
	nop
L717:
L714:
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	mov %o0,%l1
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L718
	nop
	cmp %l1,15
	ble L715
	nop
	b L712
	nop
L718:
	cmp %l1,31
	ble L715
	nop
	b L712
	nop
L719:
L715:
	mov 2,%o0
	b L713
	nop
L712:
	mov 1,%o0
L713:
	st %o0,[%fp-72]
	ld [%fp-72],%o0
	cmp %o0,2
	bne L720
	nop
	ld [%fp-20],%o0
	cmp %o0,26
	be L721
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	mov %o0,%l2
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L723
	nop
	and %l2,15,%o0
	cmp %o0,0
	bne L724
	nop
	b L722
	nop
L723:
	and %l2,31,%o0
	cmp %o0,0
	bne L724
	nop
	b L722
	nop
L724:
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	mov %o0,%l5
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L725
	nop
	add %l5,15,%l4
	b L726
	nop
L725:
	add %l5,31,%l4
L726:
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L727
	nop
	mov %l4,%l3
	cmp %l3,0
	bge L729
	nop
	add %l3,15,%l3
L729:
	sra %l3,4,%l3
	b L728
	nop
L727:
	mov %l4,%l3
	cmp %l3,0
	bge L730
	nop
	add %l3,31,%l3
L730:
	sra %l3,5,%l3
L728:
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%l6
	ld [%fp-64],%l7
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L731
	nop
	mov %l3,%o0
	sll %o0,1,%o1
	sub %o1,%l6,%o2
	add %l7,%o2,%o0
	b L732
	nop
L731:
	mov %l3,%o1
	sll %o1,2,%o2
	sub %o2,%l6,%o1
	add %l7,%o1,%o0
L732:
	st %o0,[%fp-64]
L722:
	b L733
	nop
L721:
	ld [%i3+52],%o0
	call _size_in_bytes,0
	nop
	st %o0,[%fp-76]
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L734
	nop
	mov 16,%o2
	b L735
	nop
L734:
	mov 32,%o2
L735:
	ld [%fp-76],%o0
	mov 8,%o1
	call _convert_units,0
	nop
	st %o0,[%fp-80]
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L736
	nop
	mov 16,%o1
	b L737
	nop
L736:
	mov 32,%o1
L737:
	ld [%fp-80],%o0
	mov 8,%o2
	call _convert_units,0
	nop
	st %o0,[%fp-84]
	ld [%fp-84],%o0
	st %o0,[%fp-88]
	ld [%fp-88],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L738
	nop
	ld [%fp-88],%o0
	ld [%fp-64],%o1
	ld [%o0+16],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-64]
	b L739
	nop
L738:
	ld [%fp-60],%o0
	cmp %o0,0
	bne L740
	nop
	ld [%fp-88],%o0
	st %o0,[%fp-60]
	b L741
	nop
L740:
	mov 63,%o0
	ld [%fp-60],%o1
	ld [%fp-88],%o2
	call _genop,0
	nop
	st %o0,[%fp-60]
L741:
L739:
	ld [%fp-76],%o0
	st %o0,[%fp-92]
	ld [%fp-92],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L742
	nop
	ld [%fp-92],%o0
	ld [%fp-64],%o1
	ld [%o0+16],%o0
	sub %o1,%o0,%o1
	st %o1,[%fp-64]
	b L743
	nop
L742:
	ld [%fp-60],%o0
	cmp %o0,0
	bne L744
	nop
	sethi %hi(_integer_zero_node),%o1
	mov 64,%o0
	ld [%o1+%lo(_integer_zero_node)],%o1
	ld [%fp-92],%o2
	call _genop,0
	nop
	st %o0,[%fp-60]
	b L745
	nop
L744:
	mov 64,%o0
	ld [%fp-60],%o1
	ld [%fp-92],%o2
	call _genop,0
	nop
	st %o0,[%fp-60]
L745:
L743:
L733:
L720:
	ld [%fp-60],%o0
	cmp %o0,0
	bne L746
	nop
	mov 30,%o0
	mov 0,%o1
	ld [%fp-64],%o2
	call _gen_rtx,0
	nop
	st %o0,[%fp-124]
	b L747
	nop
L746:
	ld [%fp-60],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-64],%o1
	call _plus_constant,0
	nop
	st %o0,[%fp-124]
L747:
	ld [%fp-124],%o5
	st %o5,[%fp-68]
	sethi %hi(_arg_pointer_rtx),%o2
	mov 44,%o0
	mov 4,%o1
	ld [%o2+%lo(_arg_pointer_rtx)],%o2
	ld [%fp-68],%o3
	call _gen_rtx,0
	nop
	mov %o0,%o1
	ld [%fp-20],%o0
	call _memory_address,0
	nop
	mov %o0,%o2
	mov 37,%o0
	ld [%fp-20],%o1
	call _gen_rtx,0
	nop
	mov %o0,%i5
	ld [%fp-52],%o0
	and %o0,1,%o1
	sll %o1,3,%o0
	ld [%i5],%o2
	and %o2,-9,%o1
	or %o1,%o0,%o1
	st %o1,[%i5]
	mov 0,%i4
	ld [%i3+8],%o0
	ld [%o0+24],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,35
	be L749
	nop
	ld [%fp-60],%o0
	cmp %o0,0
	bne L749
	nop
	b L748
	nop
L749:
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L750
	nop
	ld [%fp-100],%o5
	cmp %o5,7
	bg L750
	nop
	ld [%fp-100],%o0
	cmp %o0,0
	bge L752
	nop
	add %o0,3,%o0
L752:
	sra %o0,2,%o2
	mov 34,%o0
	ld [%fp-20],%o1
	call _gen_rtx,0
	nop
	mov %o0,%i4
	b L751
	nop
L750:
	mov 0,%i4
L751:
L748:
	st %g0,[%fp-92]
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,16,%o0
	cmp %o0,0
	be L753
	nop
	ld [%fp-100],%o5
	cmp %o5,7
	bg L753
	nop
	ld [%fp-20],%o0
	cmp %o0,26
	bne L755
	nop
	ld [%i3+52],%o0
	call _int_size_in_bytes,0
	nop
	ld [%fp-100],%o5
	add %o5,%o0,%o1
	cmp %o1,8
	bg L756
	nop
	b L753
	nop
L755:
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o2
	ld [%fp-100],%o5
	add %o5,%o2,%o0
	cmp %o0,8
	bg L756
	nop
	b L753
	nop
L756:
	ld [%fp-100],%o0
	cmp %o0,0
	bge L757
	nop
	add %o0,3,%o0
L757:
	sra %o0,2,%o0
	mov 2,%o1
	sub %o1,%o0,%o0
	b L754
	nop
L753:
	mov 0,%o0
L754:
	st %o0,[%fp-92]
	ld [%i3+4],%o0
	cmp %o0,0
	bne L758
	nop
	ld [%fp-48],%o0
	cmp %o0,0
	be L758
	nop
	cmp %i4,0
	be L758
	nop
	ldub [%i4+2],%o1
	and %o1,0xff,%o0
	cmp %o0,26
	bne L759
	nop
	ld [%i4],%o1
	srl %o1,8,%o0
	and %o0,255,%o1
	mov %o1,%o2
	sll %o2,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	cmp %o0,0
	bge L760
	nop
	add %o0,3,%o0
L760:
	sra %o0,2,%o0
	st %o0,[%fp-92]
	b L761
	nop
L759:
	ld [%i3+52],%o0
	call _int_size_in_bytes,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	cmp %o0,0
	bge L762
	nop
	add %o0,3,%o0
L762:
	sra %o0,2,%o0
	st %o0,[%fp-92]
L761:
L758:
	ld [%fp-92],%o0
	cmp %o0,0
	ble L763
	nop
	sethi %hi(_current_function_pretend_args_size),%o0
	ld [%fp-92],%o1
	mov %o1,%o2
	sll %o2,2,%o1
	mov %o1,%i2
	sethi %hi(_target_flags),%o1
	ld [%o1+%lo(_target_flags)],%o2
	and %o2,32,%o1
	cmp %o1,0
	be L764
	nop
	add %i2,1,%i1
	b L765
	nop
L764:
	add %i2,3,%i1
L765:
	sethi %hi(_target_flags),%o1
	ld [%o1+%lo(_target_flags)],%o2
	and %o2,32,%o1
	cmp %o1,0
	be L766
	nop
	mov %i1,%i0
	srl %i0,31,%o1
	add %i0,%o1,%i0
	sra %i0,1,%i0
	b L767
	nop
L766:
	mov %i1,%i0
	cmp %i0,0
	bge L768
	nop
	add %i0,3,%i0
L768:
	sra %i0,2,%i0
L767:
	sethi %hi(_target_flags),%o1
	ld [%o1+%lo(_target_flags)],%o2
	and %o2,32,%o1
	cmp %o1,0
	be L769
	nop
	mov %i0,%o1
	sll %o1,1,%o2
	mov %o2,%o1
	b L770
	nop
L769:
	mov %i0,%o2
	sll %o2,2,%o3
	mov %o3,%o1
L770:
	st %o1,[%o0+%lo(_current_function_pretend_args_size)]
	ld [%fp-92],%o0
	st %o0,[%fp-88]
L771:
	ld [%fp-88],%o1
	add %o1,-1,%o0
	mov %o0,%o1
	st %o1,[%fp-88]
	cmp %o1,0
	bl L772
	nop
	sethi %hi(_mode_size+16),%o1
	ld [%fp-88],%o0
	ld [%o1+%lo(_mode_size+16)],%o1
	call .umul,0
	nop
	mov %o0,%o1
	ld [%i5+4],%o0
	call _plus_constant,0
	nop
	mov %o0,%o2
	mov 37,%o0
	mov 4,%o1
	call _gen_rtx,0
	nop
	st %o0,[%fp-132]
	ld [%i4+4],%o0
	ld [%fp-88],%o1
	add %o0,%o1,%o2
	mov 34,%o0
	mov 4,%o1
	call _gen_rtx,0
	nop
	mov %o0,%o1
	ld [%fp-132],%o0
	call _emit_move_insn,0
	nop
	b L771
	nop
L772:
	mov %i5,%i4
L763:
	cmp %i4,0
	bne L773
	nop
	mov %i5,%i4
L773:
	cmp %i4,%i5
	bne L774
	nop
	ld [%fp-64],%o0
	mov %o0,%o1
	sll %o1,3,%o0
	st %o0,[%i3+44]
L774:
	cmp %i4,%i5
	bne L775
	nop
	ld [%i3+52],%o0
	call _size_in_bytes,0
	nop
	st %o0,[%fp-92]
	ld [%fp-72],%o0
	cmp %o0,0
	be L776
	nop
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L777
	nop
	mov 16,%o2
	b L778
	nop
L777:
	mov 32,%o2
L778:
	ld [%fp-92],%o0
	mov 8,%o1
	call _convert_units,0
	nop
	st %o0,[%fp-88]
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,32,%o0
	cmp %o0,0
	be L779
	nop
	mov 16,%o1
	b L780
	nop
L779:
	mov 32,%o1
L780:
	ld [%fp-88],%o0
	mov 8,%o2
	call _convert_units,0
	nop
	st %o0,[%fp-92]
L776:
	ld [%fp-92],%o0
	st %o0,[%fp-84]
	ld [%fp-84],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,35
	bne L781
	nop
	ld [%fp-84],%o0
	ld [%fp-32],%o1
	ld [%o0+16],%o0
	add %o1,%o0,%o1
	st %o1,[%fp-32]
	b L782
	nop
L781:
	ld [%fp-28],%o0
	cmp %o0,0
	bne L783
	nop
	ld [%fp-84],%o0
	st %o0,[%fp-28]
	b L784
	nop
L783:
	mov 63,%o0
	ld [%fp-28],%o1
	ld [%fp-84],%o2
	call _genop,0
	nop
	st %o0,[%fp-28]
L784:
L782:
	b L785
	nop
L775:
	mov 0,%i5
L785:
	ld [%fp-24],%o0
	cmp %o0,26
	be L786
	nop
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	cmp %o0,%o1
	be L786
	nop
	cmp %i5,0
	be L786
	nop
	ld [%fp-24],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o0
	cmp %o0,3
	bg L787
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%fp-24],%o2
	mov %o2,%o3
	sll %o3,2,%o2
	sethi %hi(_mode_size),%o4
	or %o4,%lo(_mode_size),%o3
	ld [%o0+%o1],%o0
	ld [%o2+%o3],%o1
	sub %o0,%o1,%o0
	ld [%fp-64],%o1
	add %o1,%o0,%o0
	st %o0,[%fp-64]
	ld [%fp-60],%o0
	cmp %o0,0
	bne L788
	nop
	mov 30,%o0
	mov 0,%o1
	ld [%fp-64],%o2
	call _gen_rtx,0
	nop
	st %o0,[%fp-140]
	b L789
	nop
L788:
	ld [%fp-60],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	ld [%fp-64],%o1
	call _plus_constant,0
	nop
	st %o0,[%fp-140]
L789:
	ld [%fp-140],%o5
	st %o5,[%fp-68]
L787:
	sethi %hi(_arg_pointer_rtx),%o2
	mov 44,%o0
	mov 4,%o1
	ld [%o2+%lo(_arg_pointer_rtx)],%o2
	ld [%fp-68],%o3
	call _gen_rtx,0
	nop
	mov %o0,%o1
	ld [%fp-24],%o0
	call _memory_address,0
	nop
	mov %o0,%o2
	mov 37,%o0
	ld [%fp-24],%o1
	call _gen_rtx,0
	nop
	mov %o0,%i5
	ld [%fp-52],%o0
	and %o0,1,%o1
	sll %o1,3,%o0
	ld [%i5],%o2
	and %o2,-9,%o1
	or %o1,%o0,%o1
	st %o1,[%i5]
L786:
	ld [%fp-24],%o0
	cmp %o0,26
	bne L790
	nop
	lduh [%i4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,34
	bne L791
	nop
	cmp %i5,0
	bne L792
	nop
	ld [%i4],%o1
	srl %o1,8,%o0
	and %o0,255,%o5
	st %o5,[%fp-148]
	ld [%i3+8],%o0
	call _int_size_in_bytes,0
	nop
	mov %o0,%o1
	ld [%fp-148],%o0
	call _assign_stack_local,0
	nop
	mov %o0,%i5
L792:
	ld [%i3+8],%o0
	call _int_size_in_bytes,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	cmp %o0,0
	bge L793
	nop
	add %o0,3,%o0
L793:
	sra %o0,2,%o2
	ld [%i4+4],%o0
	mov %i5,%o1
	call _move_block_from_reg,0
	nop
L791:
	st %i5,[%i3+64]
	b L794
	nop
L790:
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	be L796
	nop
	ld [%i3+12],%o0
	sethi %hi(8192),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L796
	nop
	ld [%fp+68],%o0
	ld [%o0+12],%o1
	and %o1,512,%o0
	cmp %o0,0
	bne L796
	nop
	b L795
	nop
L796:
	ld [%i3+12],%o0
	sethi %hi(16384),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L795
	nop
	ld [%i3+12],%o0
	sethi %hi(1048576),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L795
	nop
	sethi %hi(_flag_float_store),%o0
	ld [%o0+%lo(_flag_float_store)],%o1
	cmp %o1,0
	be L797
	nop
	ld [%i3+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,6
	bne L797
	nop
	b L795
	nop
L797:
	ld [%fp-24],%o0
	call _gen_reg_rtx,0
	nop
	st %o0,[%fp-156]
	ld [%fp-156],%o5
	ld [%o5],%o0
	or %o0,16,%o1
	ld [%fp-156],%o5
	st %o1,[%o5]
	ld [%fp-156],%o5
	st %o5,[%i3+64]
	ld [%fp-156],%o5
	ldub [%o5+2],%o1
	and %o1,0xff,%o0
	ldub [%i4+2],%o2
	and %o2,0xff,%o1
	cmp %o0,%o1
	be L798
	nop
	ld [%fp-156],%o0
	mov %i4,%o1
	mov 0,%o2
	call _convert_move,0
	nop
	b L799
	nop
L798:
	ld [%fp-156],%o0
	mov %i4,%o1
	call _emit_move_insn,0
	nop
L799:
	ld [%fp-156],%o5
	ld [%o5+4],%o0
	ld [%fp-44],%o1
	cmp %o0,%o1
	bl L800
	nop
	ld [%fp-156],%o5
	ld [%o5+4],%o0
	add %o0,5,%o1
	st %o1,[%fp-44]
	ld [%fp-44],%o0
	mov %o0,%o1
	sll %o1,2,%o2
	mov %o2,%o0
	call _oballoc,0
	nop
	st %o0,[%fp-84]
	sethi %hi(_parm_reg_stack_loc),%o1
	ld [%fp-44],%o0
	mov %o0,%o3
	sll %o3,2,%o2
	ld [%fp-84],%o0
	ld [%o1+%lo(_parm_reg_stack_loc)],%o1
	call _memcpy,0
	nop
	sethi %hi(_parm_reg_stack_loc),%o0
	ld [%fp-84],%o1
	st %o1,[%o0+%lo(_parm_reg_stack_loc)]
L800:
	ld [%fp-156],%o5
	ld [%o5+4],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_parm_reg_stack_loc),%o2
	ld [%o2+%lo(_parm_reg_stack_loc)],%o1
	add %o0,%o1,%o0
	st %i5,[%o0]
	ld [%fp-24],%o0
	ld [%fp-20],%o1
	cmp %o0,%o1
	bne L801
	nop
	lduh [%i4],%o0
	sll %o0,16,%o1
	srl %o1,16,%o0
	cmp %o0,37
	bne L801
	nop
	ld [%fp-60],%o0
	cmp %o0,0
	bne L801
	nop
	call _get_last_insn,0
	nop
	st %o0,[%fp-164]
	call _get_last_insn,0
	nop
	mov %o0,%o3
	mov 2,%o0
	mov 3,%o1
	mov %i4,%o2
	ld [%o3+28],%o3
	call _gen_rtx,0
	nop
	ld [%fp-164],%o5
	st %o0,[%o5+28]
L801:
	ld [%i3+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,11
	bne L802
	nop
	ld [%fp-156],%o0
	call _mark_reg_pointer,0
	nop
L802:
	b L803
	nop
L795:
	ld [%fp-20],%o0
	ld [%fp-24],%o1
	cmp %o0,%o1
	be L804
	nop
	ld [%fp-24],%o0
	mov %i4,%o1
	mov 0,%o2
	call _convert_to_mode,0
	nop
	mov %o0,%i4
L804:
	cmp %i4,%i5
	be L805
	nop
	cmp %i5,0
	bne L806
	nop
	ld [%i4],%o0
	srl %o0,8,%o1
	and %o1,255,%o0
	ld [%i4],%o2
	srl %o2,8,%o1
	and %o1,255,%o2
	mov %o2,%o3
	sll %o3,2,%o1
	sethi %hi(_mode_size),%o3
	or %o3,%lo(_mode_size),%o2
	ld [%o1+%o2],%o1
	call _assign_stack_local,0
	nop
	mov %o0,%i5
L806:
	mov %i5,%o0
	mov %i4,%o1
	call _emit_move_insn,0
	nop
L805:
	st %i5,[%i3+64]
	sethi %hi(_frame_pointer_needed),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_frame_pointer_needed)]
L803:
L794:
	ld [%i3+12],%o0
	sethi %hi(1048576),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	be L807
	nop
	ld [%i3+64],%o0
	ld [%o0],%o1
	or %o1,16,%o2
	st %o2,[%o0]
L807:
	ld [%i3+12],%o0
	sethi %hi(262144),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	be L808
	nop
	ld [%i3+64],%o0
	ld [%o0],%o1
	or %o1,32,%o2
	st %o2,[%o0]
L808:
	ld [%fp-20],%o0
	cmp %o0,26
	be L809
	nop
	ld [%fp-20],%o0
	mov %o0,%o1
	sll %o1,2,%o0
	sethi %hi(_mode_size),%o2
	or %o2,%lo(_mode_size),%o1
	ld [%o0+%o1],%o2
	add %o2,3,%o0
	and %o0,-4,%o1
	ld [%fp-100],%o5
	add %o5,%o1,%o5
	st %o5,[%fp-100]
	b L810
	nop
L809:
	ld [%i3+52],%o0
	call _int_size_in_bytes,0
	nop
	add %o0,3,%o1
	and %o1,-4,%o0
	ld [%fp-100],%o5
	add %o5,%o0,%o5
	st %o5,[%fp-100]
L810:
L707:
	ld [%i3+4],%i3
	b L705
	nop
L706:
	call _max_reg_num,0
	nop
	sethi %hi(_max_parm_reg),%o1
	st %o0,[%o1+%lo(_max_parm_reg)]
	call _get_last_insn,0
	nop
	sethi %hi(_last_parm_insn),%o1
	st %o0,[%o1+%lo(_last_parm_insn)]
	sethi %hi(_current_function_args_size),%o0
	ld [%fp-32],%o1
	st %o1,[%o0+%lo(_current_function_args_size)]
L700:
	ret
	restore
	.align 4
	.global _get_structure_value_addr
	.proc	0110
_get_structure_value_addr:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,30
	be L812
	nop
	call _abort,0
	nop
L812:
	ld [%fp+68],%o0
	ld [%o0+4],%l0
	add %l0,1,%l0
	srl %l0,31,%o0
	add %l0,%o0,%l0
	sra %l0,1,%l0
	mov %l0,%o0
	sll %o0,1,%o1
	mov %o1,%l0
	sethi %hi(_max_structure_value_size),%o0
	ld [%o0+%lo(_max_structure_value_size)],%o1
	cmp %l0,%o1
	ble L813
	nop
	sethi %hi(_max_structure_value_size),%o0
	st %l0,[%o0+%lo(_max_structure_value_size)]
	mov 26,%o0
	mov %l0,%o1
	call _assign_stack_local,0
	nop
	sethi %hi(_structure_value),%o1
	st %o0,[%o1+%lo(_structure_value)]
	sethi %hi(_structure_value),%o1
	ld [%o1+%lo(_structure_value)],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,37
	bne L814
	nop
	sethi %hi(_structure_value),%o0
	sethi %hi(_structure_value),%o2
	ld [%o2+%lo(_structure_value)],%o1
	ld [%o1+4],%o2
	st %o2,[%o0+%lo(_structure_value)]
L814:
L813:
	sethi %hi(_structure_value),%o0
	ld [%o0+%lo(_structure_value)],%i0
	b L811
	nop
L811:
	ret
	restore
	.align 8
LC14:
	.ascii "`%s' may be used uninitialized in this function\0"
	.align 8
LC15:
	.ascii "variable `%s' may be clobbered by `longjmp'\0"
	.align 4
	.global _uninitialized_vars_warning
	.proc	020
_uninitialized_vars_warning:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+28],%l0
L816:
	cmp %l0,0
	be L817
	nop
	ldub [%l0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	bne L819
	nop
	ld [%l0+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,19
	be L819
	nop
	ld [%l0+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,20
	be L819
	nop
	ld [%l0+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,16
	be L819
	nop
	ld [%l0+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L819
	nop
	ld [%l0+64],%o1
	ld [%o1+4],%o0
	call _regno_uninitialized,0
	nop
	cmp %o0,0
	be L819
	nop
	mov %l0,%o0
	sethi %hi(LC14),%o2
	or %o2,%lo(LC14),%o1
	call _warning_with_decl,0
	nop
L819:
	ldub [%l0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	bne L820
	nop
	ld [%l0+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L820
	nop
	ld [%l0+64],%o1
	ld [%o1+4],%o0
	call _regno_clobbered_at_setjmp,0
	nop
	cmp %o0,0
	be L820
	nop
	mov %l0,%o0
	sethi %hi(LC15),%o2
	or %o2,%lo(LC15),%o1
	call _warning_with_decl,0
	nop
L820:
L818:
	ld [%l0+4],%l0
	b L816
	nop
L817:
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%l1
L821:
	cmp %l1,0
	be L822
	nop
	mov %l1,%o0
	call _uninitialized_vars_warning,0
	nop
L823:
	ld [%l1+4],%l1
	b L821
	nop
L822:
L815:
	ret
	restore
	.align 4
	.global _setjmp_protect
	.proc	020
_setjmp_protect:
	!#PROLOGUE# 0
	save %sp,-112,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	ld [%fp+68],%o0
	ld [%o0+28],%l0
L825:
	cmp %l0,0
	be L826
	nop
	ldub [%l0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,43
	be L829
	nop
	ldub [%l0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,44
	be L829
	nop
	b L828
	nop
L829:
	ld [%l0+64],%o0
	cmp %o0,0
	be L828
	nop
	ld [%l0+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L828
	nop
	ld [%l0+12],%o0
	sethi %hi(8192),%o1
	and %o0,%o1,%o0
	cmp %o0,0
	bne L828
	nop
	mov %l0,%o0
	call _put_var_into_stack,0
	nop
L828:
L827:
	ld [%l0+4],%l0
	b L825
	nop
L826:
	nop
	ld [%fp+68],%o0
	ld [%o0+24],%l1
L830:
	cmp %l1,0
	be L831
	nop
	mov %l1,%o0
	call _setjmp_protect,0
	nop
L832:
	ld [%l1+4],%l1
	b L830
	nop
L831:
L824:
	ret
	restore
	.align 4
	.global _expand_function_start
	.proc	020
_expand_function_start:
	!#PROLOGUE# 0
	save %sp,-120,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	sethi %hi(_this_function),%o0
	ld [%fp+68],%o1
	st %o1,[%o0+%lo(_this_function)]
	sethi %hi(_cse_not_expected),%o0
	sethi %hi(_optimize),%o1
	ld [%o1+%lo(_optimize)],%o2
	xor %o2,0,%o3
	subcc %g0,%o3,%g0
	subx %g0,-1,%o1
	st %o1,[%o0+%lo(_cse_not_expected)]
	sethi %hi(_frame_pointer_needed),%o0
	sethi %hi(_flag_omit_frame_pointer),%o1
	ld [%o1+%lo(_flag_omit_frame_pointer)],%o2
	xor %o2,0,%o3
	subcc %g0,%o3,%g0
	subx %g0,-1,%o1
	st %o1,[%o0+%lo(_frame_pointer_needed)]
	sethi %hi(_goto_fixup_chain),%o0
	st %g0,[%o0+%lo(_goto_fixup_chain)]
	sethi %hi(_stack_slot_list),%o0
	st %g0,[%o0+%lo(_stack_slot_list)]
	sethi %hi(_invalid_stack_slot),%o0
	st %g0,[%o0+%lo(_invalid_stack_slot)]
	sethi %hi(_write_symbols),%o1
	ld [%o1+%lo(_write_symbols)],%o0
	call _init_emit,0
	nop
	call _init_expr,0
	nop
	call _init_const_rtx_hash_table,0
	nop
	sethi %hi(_current_function_pops_args),%l1
	mov 0,%l2
	sethi %hi(_target_flags),%o0
	ld [%o0+%lo(_target_flags)],%o1
	and %o1,8,%o0
	cmp %o0,0
	be L834
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ldub [%o1+12],%o2
	and %o2,0xff,%o0
	cmp %o0,1
	be L834
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ld [%o1+16],%o0
	cmp %o0,0
	be L835
	nop
	ld [%fp+68],%o0
	ld [%o0+8],%o1
	ld [%o1+16],%o0
	call _tree_last,0
	nop
	sethi %hi(_void_type_node),%o1
	ld [%o0+20],%o0
	ld [%o1+%lo(_void_type_node)],%o1
	cmp %o0,%o1
	be L835
	nop
	b L834
	nop
L835:
	mov 1,%l2
L834:
	st %l2,[%l1+%lo(_current_function_pops_args)]
	sethi %hi(_current_function_name),%o0
	ld [%fp+68],%o1
	ld [%o1+36],%o2
	ld [%o2+20],%o1
	st %o1,[%o0+%lo(_current_function_name)]
	sethi %hi(_current_function_needs_context),%o0
	mov 0,%o1
	sethi %hi(_current_function_decl),%o3
	ld [%o3+%lo(_current_function_decl)],%o2
	ld [%o2+40],%o3
	cmp %o3,0
	be L836
	nop
	sethi %hi(_current_function_decl),%o3
	ld [%o3+%lo(_current_function_decl)],%o2
	ld [%o2+40],%o3
	ldub [%o3+12],%o4
	and %o4,0xff,%o2
	cmp %o2,28
	bne L836
	nop
	mov 1,%o1
L836:
	st %o1,[%o0+%lo(_current_function_needs_context)]
	sethi %hi(_current_function_calls_setjmp),%o0
	st %g0,[%o0+%lo(_current_function_calls_setjmp)]
	sethi %hi(_current_function_returns_pcc_struct),%o0
	st %g0,[%o0+%lo(_current_function_returns_pcc_struct)]
	sethi %hi(_current_function_returns_struct),%o0
	st %g0,[%o0+%lo(_current_function_returns_struct)]
	sethi %hi(_max_structure_value_size),%o0
	st %g0,[%o0+%lo(_max_structure_value_size)]
	sethi %hi(_structure_value),%o0
	st %g0,[%o0+%lo(_structure_value)]
	sethi %hi(_block_stack),%o0
	st %g0,[%o0+%lo(_block_stack)]
	sethi %hi(_loop_stack),%o0
	st %g0,[%o0+%lo(_loop_stack)]
	sethi %hi(_case_stack),%o0
	st %g0,[%o0+%lo(_case_stack)]
	sethi %hi(_cond_stack),%o0
	st %g0,[%o0+%lo(_cond_stack)]
	sethi %hi(_nesting_stack),%o0
	st %g0,[%o0+%lo(_nesting_stack)]
	sethi %hi(_nesting_depth),%o0
	st %g0,[%o0+%lo(_nesting_depth)]
	sethi %hi(_tail_recursion_label),%o0
	st %g0,[%o0+%lo(_tail_recursion_label)]
	sethi %hi(_frame_offset),%o0
	st %g0,[%o0+%lo(_frame_offset)]
	sethi %hi(_save_expr_regs),%o0
	st %g0,[%o0+%lo(_save_expr_regs)]
	sethi %hi(_rtl_expr_chain),%o0
	st %g0,[%o0+%lo(_rtl_expr_chain)]
	sethi %hi(_immediate_size_expand),%o1
	sethi %hi(_immediate_size_expand),%o0
	sethi %hi(_immediate_size_expand),%o1
	ld [%o1+%lo(_immediate_size_expand)],%o2
	add %o2,1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_immediate_size_expand)]
	call _init_pending_stack_adjust,0
	nop
	call _clear_current_args_size,0
	nop
	sethi %hi(_current_function_pretend_args_size),%o0
	st %g0,[%o0+%lo(_current_function_pretend_args_size)]
	ld [%fp+68],%o0
	ld [%fp+68],%o1
	ld [%o0+16],%o0
	ld [%o1+20],%o1
	call _emit_line_note,0
	nop
	mov 0,%o0
	mov -1,%o1
	call _emit_note,0
	nop
	ld [%fp+68],%o0
	call _assign_parms,0
	nop
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+28],%o0
	cmp %o0,26
	be L838
	nop
	sethi %hi(_flag_pcc_struct_return),%o0
	ld [%o0+%lo(_flag_pcc_struct_return)],%o1
	cmp %o1,0
	be L837
	nop
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,19
	be L838
	nop
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	ldub [%o0+12],%o1
	and %o1,0xff,%o0
	cmp %o0,20
	be L838
	nop
	b L837
	nop
L838:
	sethi %hi(_flag_pcc_struct_return),%o0
	ld [%o0+%lo(_flag_pcc_struct_return)],%o1
	cmp %o1,0
	be L839
	nop
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	call _int_size_in_bytes,0
	nop
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	call _assemble_static_space,0
	nop
	mov %o0,%l1
	sethi %hi(_current_function_returns_pcc_struct),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_current_function_returns_pcc_struct)]
	b L840
	nop
L839:
	mov 4,%o0
	call _gen_reg_rtx,0
	nop
	mov %o0,%l1
	sethi %hi(_struct_value_incoming_rtx),%o1
	mov %l1,%o0
	ld [%o1+%lo(_struct_value_incoming_rtx)],%o1
	call _emit_move_insn,0
	nop
	sethi %hi(_current_function_returns_struct),%o0
	mov 1,%o1
	st %o1,[%o0+%lo(_current_function_returns_struct)]
L840:
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	mov 37,%o0
	ld [%o1+28],%o1
	mov %l1,%o2
	call _gen_rtx,0
	nop
	ld [%fp+68],%o1
	ld [%o1+56],%o2
	st %o0,[%o2+64]
	b L841
	nop
L837:
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	ldub [%o0+28],%o2
	and %o2,0xff,%o1
	mov 34,%o0
	mov 0,%o2
	call _gen_rtx,0
	nop
	ld [%fp+68],%o1
	ld [%o1+56],%o2
	st %o0,[%o2+64]
L841:
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	lduh [%o0],%o1
	sll %o1,16,%o2
	srl %o2,16,%o0
	cmp %o0,34
	bne L842
	nop
	ld [%fp+68],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	ld [%o0],%o1
	or %o1,2,%o2
	st %o2,[%o0]
L842:
	b L843
	nop
	sethi %hi(_return_label),%o0
	st %g0,[%o0+%lo(_return_label)]
	b L844
	nop
L843:
	call _gen_label_rtx,0
	nop
	sethi %hi(_return_label),%o1
	st %o0,[%o1+%lo(_return_label)]
L844:
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	be L845
	nop
	call _get_last_insn,0
	nop
	sethi %hi(_parm_birth_insn),%o1
	st %o0,[%o1+%lo(_parm_birth_insn)]
	mov 56,%l0
L846:
	sethi %hi(_max_parm_reg),%o0
	ld [%o0+%lo(_max_parm_reg)],%o1
	cmp %l0,%o1
	bge L847
	nop
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_regno_reg_rtx),%o1
	ld [%o1+%lo(_regno_reg_rtx)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	call _use_variable,0
	nop
L848:
	add %l0,1,%l0
	b L846
	nop
L847:
L845:
	call _get_last_insn,0
	nop
	sethi %hi(_tail_recursion_reentry),%o1
	st %o0,[%o1+%lo(_tail_recursion_reentry)]
	call _get_pending_sizes,0
	nop
	st %o0,[%fp-20]
L849:
	ld [%fp-20],%o0
	cmp %o0,0
	be L850
	nop
	ld [%fp-20],%o1
	ld [%o1+20],%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o3
	call _expand_expr,0
	nop
L851:
	ld [%fp-20],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	b L849
	nop
L850:
L833:
	ret
	restore
	.align 4
	.global _expand_function_end
	.proc	020
_expand_function_end:
	!#PROLOGUE# 0
	save %sp,-136,%sp
	!#PROLOGUE# 1
	st %i0,[%fp+68]
	st %i1,[%fp+72]
L853:
	sethi %hi(_sequence_stack),%o0
	ld [%o0+%lo(_sequence_stack)],%o1
	cmp %o1,0
	be L854
	nop
	mov 0,%o0
	call _end_sequence,0
	nop
	b L853
	nop
L854:
	sethi %hi(_immediate_size_expand),%o1
	sethi %hi(_immediate_size_expand),%o0
	sethi %hi(_immediate_size_expand),%o1
	ld [%o1+%lo(_immediate_size_expand)],%o2
	add %o2,-1,%o1
	mov %o1,%o2
	st %o2,[%o0+%lo(_immediate_size_expand)]
	sethi %hi(_current_function_returns_struct),%o0
	ld [%o0+%lo(_current_function_returns_struct)],%o1
	cmp %o1,0
	be L855
	nop
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-20]
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	st %o0,[%fp-24]
	ld [%fp-24],%o0
	call _build_pointer_type,0
	nop
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o1
	call _hard_function_value,0
	nop
	st %o0,[%fp-28]
	ld [%fp-28],%o0
	ld [%fp-20],%o1
	call _emit_move_insn,0
	nop
L855:
	sethi %hi(_obey_regdecls),%o0
	ld [%o0+%lo(_obey_regdecls)],%o1
	cmp %o1,0
	be L856
	nop
	mov 56,%l0
L857:
	sethi %hi(_max_parm_reg),%o0
	ld [%o0+%lo(_max_parm_reg)],%o1
	cmp %l0,%o1
	bge L858
	nop
	mov %l0,%o1
	sll %o1,2,%o0
	sethi %hi(_regno_reg_rtx),%o1
	ld [%o1+%lo(_regno_reg_rtx)],%o2
	add %o0,%o2,%o1
	ld [%o1],%o0
	call _use_variable,0
	nop
L859:
	add %l0,1,%l0
	b L857
	nop
L858:
	nop
	sethi %hi(_save_expr_regs),%o0
	ld [%o0+%lo(_save_expr_regs)],%o1
	st %o1,[%fp-32]
L860:
	ld [%fp-32],%o0
	cmp %o0,0
	be L861
	nop
	ld [%fp-32],%o1
	ld [%o1+4],%o0
	call _use_variable,0
	nop
	ld [%fp-32],%o0
	sethi %hi(_parm_birth_insn),%o1
	ld [%o0+4],%o0
	ld [%o1+%lo(_parm_birth_insn)],%o1
	call _use_variable_after,0
	nop
L862:
	ld [%fp-32],%o0
	ld [%o0+8],%o1
	st %o1,[%fp-32]
	b L860
	nop
L861:
L856:
	call _clear_pending_stack_adjust,0
	nop
	call _do_pending_stack_adjust,0
	nop
	mov 0,%o0
	mov -6,%o1
	call _emit_note,0
	nop
	ld [%fp+68],%o0
	ld [%fp+72],%o1
	call _emit_line_note_force,0
	nop
	b L863
	nop
	call _gen_return,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	b L864
	nop
L863:
	sethi %hi(_return_label),%o1
	ld [%o1+%lo(_return_label)],%o0
	call _emit_label,0
	nop
L864:
	sethi %hi(_current_function_returns_pcc_struct),%o0
	ld [%o0+%lo(_current_function_returns_pcc_struct)],%o1
	cmp %o1,0
	be L865
	nop
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o0
	ld [%o0+56],%o1
	ld [%o1+64],%o0
	ld [%o0+4],%o1
	st %o1,[%fp-32]
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o0
	ld [%o0+56],%o1
	ld [%o1+8],%o0
	st %o0,[%fp-36]
	ld [%fp-36],%o0
	call _build_pointer_type,0
	nop
	sethi %hi(_current_function_decl),%o1
	ld [%o1+%lo(_current_function_decl)],%o1
	call _hard_function_value,0
	nop
	st %o0,[%fp-40]
	ld [%fp-40],%o0
	ld [%fp-32],%o1
	call _emit_move_insn,0
	nop
	ld [%fp-40],%o0
	call _use_variable,0
	nop
	b L866
	nop
	call _gen_return,0
	nop
	mov %o0,%o1
	mov %o1,%o0
	call _emit_jump_insn,0
	nop
	call _emit_barrier,0
	nop
L866:
L865:
	call _get_insns,0
	nop
	mov %o0,%o3
	mov 0,%o0
	mov 0,%o1
	mov 0,%o2
	mov 0,%o4
	call _fixup_gotos,0
	nop
L852:
	ret
	restore
	.global _current_function_calls_setjmp
	.common _current_function_calls_setjmp,8,"bss"
	.global _save_expr_regs
	.common _save_expr_regs,8,"bss"
	.global _current_function_pops_args
	.common _current_function_pops_args,8,"bss"
	.global _current_function_returns_struct
	.common _current_function_returns_struct,8,"bss"
	.global _current_function_returns_pcc_struct
	.common _current_function_returns_pcc_struct,8,"bss"
	.global _current_function_needs_context
	.common _current_function_needs_context,8,"bss"
	.global _current_function_args_size
	.common _current_function_args_size,8,"bss"
	.global _current_function_pretend_args_size
	.common _current_function_pretend_args_size,8,"bss"
	.global _current_function_name
	.common _current_function_name,8,"bss"
	.global _return_label
	.common _return_label,8,"bss"
	.global _stack_slot_list
	.common _stack_slot_list,8,"bss"
	.global _emit_filename
	.common _emit_filename,8,"bss"
	.global _emit_lineno
	.common _emit_lineno,8,"bss"

	.reserve _parm_birth_insn,8,"bss"

	.reserve _this_function,8,"bss"

	.reserve _frame_offset,8,"bss"

	.reserve _invalid_stack_slot,8,"bss"

	.reserve _tail_recursion_label,8,"bss"

	.reserve _tail_recursion_reentry,8,"bss"

	.reserve _last_expr_type,8,"bss"

	.reserve _last_expr_value,8,"bss"

	.reserve _rtl_expr_chain,8,"bss"

	.reserve _last_parm_insn,8,"bss"
	.global _block_stack
	.common _block_stack,8,"bss"
	.global _stack_block_stack
	.common _stack_block_stack,8,"bss"
	.global _cond_stack
	.common _cond_stack,8,"bss"
	.global _loop_stack
	.common _loop_stack,8,"bss"
	.global _case_stack
	.common _case_stack,8,"bss"
	.global _nesting_stack
	.common _nesting_stack,8,"bss"
	.global _nesting_depth
	.common _nesting_depth,8,"bss"

	.reserve _goto_fixup_chain,8,"bss"
	.global _expr_stmts_for_value
	.common _expr_stmts_for_value,8,"bss"

	.reserve _max_parm_reg,8,"bss"

	.reserve _parm_reg_stack_loc,8,"bss"

	.reserve _max_structure_value_size,8,"bss"

	.reserve _structure_value,8,"bss"
